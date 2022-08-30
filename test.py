# ---------------------------------------------------------------------------
# Learning Lane Graph Representations for Motion Forecasting
#
# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Written by Ming Liang, Yun Chen
# ---------------------------------------------------------------------------

import argparse
import os
import yaml
import pickle
import sys
from models.lanegcn import Net
import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from utils.general import increment_path, save_ckpt,ROOT
from data import create_dataloader
from utils.torch_utils import select_device,to_device, load_pretrain

os.umask(0)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def main(args):
    assert os.path.exists(args.model_config)
    assert os.path.exists(args.data_config)
    with open(args.model_config,"r",encoding="utf-8") as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    with open(args.data_config,"r",encoding="utf-8") as f:
        data_config = yaml.load(f,Loader=yaml.FullLoader)

    device = select_device(args.device,rank = LOCAL_RANK, batch_size=config["batch_size"]) #兜底条款

    model = Net(config).to(device)
    # load pretrain model
    ckpt_path = args.weight
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(config["save_dir"], ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    load_pretrain(model, ckpt["state_dict"])
    model.eval()

    # Data loader for evaluation
    test_loader,test_dataset= create_dataloader(
        data_type = args.data_type,
        batch_size = config["batch_size"]//4,
        workers = config["workers"],
        rank=LOCAL_RANK,
        shuffle = False,
        config = data_config
    )

    # begin inference
    preds = {}
    gts = {}
    cities = {}
    for ii, data in tqdm(enumerate(test_loader)):
        data = dict(to_device(data,device=device))
        with torch.no_grad():
            output = model(data)
            results = [x[0:1].detach().cpu().numpy() for x in output["reg"]]
        for i, (argo_idx, pred_traj) in enumerate(zip(data["argo_id"], results)):
            preds[argo_idx] = pred_traj.squeeze()
            cities[argo_idx] = data["city"][i]
            gts[argo_idx] = data["gt_preds"][i][0] if "gt_preds" in data else None

    # save for further visualization
    res = dict(
        preds = preds,
        gts = gts,
        cities = cities,
    )
    # torch.save(res,f"{config['save_dir']}/results.pkl")
    
    # evaluate or submit
    if args.data_type in ["val","test"] :
        # for val set: compute metric
        from argoverse.evaluation.eval_forecasting import (compute_forecasting_metrics,)
        # Max #guesses (K): 6
        _ = compute_forecasting_metrics(preds, gts, cities, 6, 30, 2)
        # Max #guesses (K): 1
        _ = compute_forecasting_metrics(preds, gts, cities, 1, 30, 2)
    if args.data_type in ["test"]:
        # for test set: save as h5 for submission in evaluation server
        from argoverse.evaluation.competition_util import generate_forecasting_h5
        generate_forecasting_h5(preds, f"{config['save_dir']}/submit.h5")  # this might take awhile
    # import ipdb;ipdb.set_trace()


if __name__ == "__main__":
    # define parser
    parser = argparse.ArgumentParser(description="Argoverse Motion Forecasting in Pytorch")
    parser.add_argument("--eval", action="store_true", default=True)
    parser.add_argument("--device", type = str,default = "cuda:0")
    parser.add_argument("--model_config",type=str,default=f"{str(ROOT)}/config/model.yaml")
    parser.add_argument("--data_config",type=str,default=f"{str(ROOT)}/config/data.yaml")
    parser.add_argument("--data_type", type=str, default="val", help='val or test')
    parser.add_argument("--weight", default=f"{str(ROOT)}/results/exp/36.ckpt", type=str, metavar="WEIGHT", help="checkpoint path")
    args = parser.parse_args()

    main(args)
