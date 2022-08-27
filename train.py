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
from data import ArgoDataset, collate_fn
from utils.log import Logger
from utils.torch_utils import gpu, to_long, load_pretrain,select_device,to_device
from utils.optim import Optimizer, StepLR
from utils.general import worker_init_fn, increment_path, save_ckpt,set_logging

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)

LOGGER = logging.getLogger(__name__)
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

def main(args):
    # check
    set_logging(RANK)
    assert os.path.exists(args.model_config)
    assert os.path.exists(args.data_config)
    with open(args.model_config,"r",encoding="utf-8") as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    with open(args.data_config,"r",encoding="utf-8") as f:
        data_config = yaml.load(f,Loader=yaml.FullLoader)
    
    # seed = 0
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)

    # torch.cuda.set_device(local_rank)
    # dist.init_process_group(backend='nccl')

    # torch.cuda.set_device(args.local_rank)
    # dist.init_process_group(backend='nccl')
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## DDP mode
    device = select_device(args.device, batch_size=config["batch_size"])
    if LOCAL_RANK != -1:
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        assert config["batch_size"] % WORLD_SIZE == 0, '--batch-size must be multiple of CUDA device count'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    
    if RANK in [-1,0]:
        # Logger
        save_dir = increment_path(Path(config["save_dir"]),mkdir = True)
        log_dir = save_dir / "log"
        if not log_dir.exists():
            log_dir.mkdir(parents = True)
        sys.stdout = Logger(log_dir / "log.txt")

    train_data = ArgoDataset("train",data_config["preprocess_path"],args.save_type,data_config)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    train_dataloader = DataLoader(
        train_data,
        batch_size = config["batch_size"]//WORLD_SIZE,
        num_workers = config["workers"],
        shuffle = True,
        pin_memory=True,
        collate_fn = collate_fn,
        worker_init_fn=worker_init_fn,
        sampler= train_sampler,
        drop_last=True,
    )

    val_data = ArgoDataset("val",data_config["preprocess_path"],args.save_type,data_config)
    val_sampler = DistributedSampler(val_data)
    val_dataloader = DataLoader(
        val_data,
        batch_size = config["batch_size"]//WORLD_SIZE,
        num_workers = config["val_workers"],
        shuffle = False,
        pin_memory=True,
        collate_fn = collate_fn,
        sampler=val_sampler
    )

    model = Net(config).to(device)
    model = DDP(model, device_ids=[LOCAL_RANK])
    loss = Loss(config)#.to(device)
    post_process = PostProcess(config)#.to(device)
    
    params = model.parameters()
    opt = Optimizer(params,config)

    if args.resume or args.weight:
        #TODO
        ckpt_path = args.resume or args.weight
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(config["save_dir"], ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        load_pretrain(model, ckpt["state_dict"])
        if args.resume:
            config["start_epoch"] = ckpt["start_epoch"]
            opt.load_state_dict(ckpt["opt_state"])

    metrics = dict()
    for epoch in range(config["start_epoch"], config["epochs"]+1):
        model.train()
        
        train_dataloader.sampler.set_epoch(int(epoch))
        val_dataloader.sampler.set_epoch(int(epoch))

        num_batches = len(train_dataloader)
        epoch_per_batch = 1.0 / num_batches
        save_iters = int(np.ceil(config["save_freq"] * num_batches))
        display_iters = int(config["display_iters"]*config["batch_size"]/WORLD_SIZE)

        start_time = time.time()
        for i, data in enumerate(train_dataloader):
            epoch += epoch_per_batch
            data = dict(to_device(data))

            output =  model(data)
            loss_out = loss(output,data)
            post_out = post_process(output,data)
            post_process.append(metrics,loss_out,post_out)

            opt.zero_grad()
            loss_out["loss"].backward()
            lr = opt.step(epoch)

            if RANK in [-1,0]:
                # save checkpoint
                num_iters = int(np.round(epoch * num_batches))
                if num_iters % save_iters == 0 or epoch >= config["epochs"]:
                    save_ckpt(model, opt, save_dir, epoch)
                if num_iters % display_iters == 0:
                    dt = time.time() - start_time
                    post_process.display(metrics, dt, epoch, lr)
                    start_time = time.time()
                    metrics = dict()
        model.eval()
        start_time = time.time()
        metrics = dict()
        for i,data in enumerate(val_dataloader):
            data = dict(to_device(data))
            with torch.no_grad():
                output = model(data)
                loss_out = loss(output,data)
                post_out = post_process(output, data)
                post_process.append(metrics, loss_out, post_out)
        dt = time.time() - start_time
        if RANK in [-1,0]:
            post_process.display(metrics, dt, epoch)

    if WORLD_SIZE > 1 and RANK == 0:
        LOGGER.info('Destroying process group... ')
        dist.destroy_process_group()


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Fuse Detection in Pytorch")
    parser.add_argument("-m", "--model", default="lanegcn", type=str, metavar="MODEL", help="model name")
    parser.add_argument("--eval", action = "store_true")
    parser.add_argument("--model_config",type=str,default="./config/model.yaml")
    parser.add_argument("--data_config",type=str,default="./config/data.yaml")
    parser.add_argument("--save_type",type=str, default="together") # together or apart
    parser.add_argument("--resume", default="", type=str, metavar="RESUME", help="checkpoint path")
    parser.add_argument("--weight", default="", type=str, metavar="WEIGHT", help="checkpoint path")

    args = parser.parse_args()

    main(args)