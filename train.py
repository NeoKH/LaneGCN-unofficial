import argparse
import logging
import yaml
from tqdm import tqdm
import numpy as np
import random
import os
import sys
import time
import shutil
from pathlib import Path
from importlib import import_module
from numbers import Number

from fractions import gcd
import torch
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.utils.data import Sampler, DataLoader
from torch.utils.data.distributed import DistributedSampler

from models.lanegcn import Net, Loss, PostProcess
from data import ArgoDataset,create_dataloader
from utils.log import Logger
from utils.torch_utils import gpu, to_long, load_pretrain,select_device,to_device,init_seeds
from utils.data import collate_fn
from utils.optim import Optimizer, StepLR
from utils.general import increment_path, save_ckpt,set_logging,ROOT

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

def main(args):
    # check path
    assert os.path.exists(args.model_config)
    assert os.path.exists(args.data_config)
    with open(args.model_config,"r",encoding="utf-8") as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    with open(args.data_config,"r",encoding="utf-8") as f:
        data_config = yaml.load(f,Loader=yaml.FullLoader)
    
    if RANK in [-1,0]:
        # Logger
        save_dir = increment_path(Path(config["save_dir"]),mkdir = True)
        log_dir = save_dir / "log"
        if not log_dir.exists():
            log_dir.mkdir(parents = True)
        sys.stdout = Logger(log_dir / "log.txt")
    
    init_seeds(1+RANK)

    ## DDP mode
    device = select_device(args.device,rank = LOCAL_RANK, batch_size=config["batch_size"]) #兜底条款
    if LOCAL_RANK != -1:
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        assert config["batch_size"] % WORLD_SIZE == 0, '--batch-size must be multiple of CUDA device count'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")
    cuda = device.type != 'cpu'
    model = Net(config).to(device)
    if cuda and RANK != -1:
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)
    loss = Loss(config).to(device)
    post_process = PostProcess(config).to(device)
    
    params = model.parameters()
    opt = Optimizer(params,config)

    # if args.resume or args.weight:
    #     #TODO
    #     ckpt_path = args.resume or args.weight
    #     if not os.path.isabs(ckpt_path):
    #         ckpt_path = os.path.join(config["save_dir"], ckpt_path)
    #     ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    #     load_pretrain(model, ckpt["state_dict"])
    #     if args.resume:
    #         config["start_epoch"] = ckpt["start_epoch"]
    #         opt.load_state_dict(ckpt["opt_state"])
    
    train_loader,train_dataset = create_dataloader(
        data_type = "train",
        batch_size = config["batch_size"]//WORLD_SIZE,
        workers = config["workers"],
        rank = LOCAL_RANK,
        shuffle = True,
        config = data_config
    )
    num_batches = len(train_loader)# number of batches
    if RANK in [-1,0]:
        val_loader, _ = create_dataloader(
            data_type = "val",
            batch_size = config["batch_size"]//WORLD_SIZE * 2 ,
            workers = config["workers"],
            rank = -1, # single GPU or cpu
            shuffle = False,
            config = data_config
        )
        print("Start training !")
    metrics = dict()
    for epoch in range(config["start_epoch"], config["epochs"]+1):
        model.train()
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)

        # if RANK in [-1, 0]: # TODO
        #     pbar = tqdm(pbar, total=nb)  # progress bar
        
        # epoch_per_batch = 1.0 / num_batches
        # save_iters = int(np.ceil(config["save_freq"] * num_batches))
        # display_iters = int(config["display_iters"]*config["batch_size"]/WORLD_SIZE)

        start_time = time.time()
        for i, data in enumerate(train_loader):
            data = dict(to_device(data,device=device))

            output =  model(data)
            loss_out = loss(output,data)
            opt.zero_grad()
            loss_out["loss"].backward()
            lr = opt.step(epoch)

            post_out = post_process(output,data)
            post_process.append(metrics,loss_out,post_out)

            if RANK in [-1,0]:
                # Display metrics
                if epoch*num_batches+i % config["display_iters"] == 0:
                    delta_time = time.time() - start_time
                    post_process.display(metrics, delta_time, epoch, i, lr, model_type="train")
                    metrics = dict()
        
        if RANK in [-1, 0]:
            model.eval()
            metrics = dict()
            for i,data in enumerate(val_loader):
                data = dict(to_device(data,device=device))
                with torch.no_grad():
                    output = model(data)
                    loss_out = loss(output,data)
                    post_out = post_process(output, data)
                    post_process.append(metrics, loss_out, post_out)
            delta_time = time.time() - start_time
            post_process.display(metrics, delta_time, epoch, model_type="val")
        
            # save checkpoint
            if epoch > 0 or epoch >= config["epochs"]:
                save_ckpt(model, opt, save_dir, epoch,ddp=True)

    torch.cuda.empty_cache()
    if WORLD_SIZE > 1 and RANK == 0:
        dist.destroy_process_group()
    


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Fuse Detection in Pytorch")
    parser.add_argument("-m", "--model", default="lanegcn", type=str, metavar="MODEL", help="model name")
    parser.add_argument("--eval", action = "store_true")
    parser.add_argument("--device", type = str,default = "cpu")
    parser.add_argument("--model_config",type=str,default=f"{str(ROOT)}/config/model.yaml")
    parser.add_argument("--data_config",type=str,default=f"{str(ROOT)}/config/data.yaml")
    parser.add_argument("--resume", default="", type=str, metavar="RESUME", help="checkpoint path")
    parser.add_argument("--weight", default="", type=str, metavar="WEIGHT", help="checkpoint path")

    args = parser.parse_args()

    main(args)