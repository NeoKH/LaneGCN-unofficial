#!/usr/bin/env python3
# coding:utf8
import os
import yaml
import copy
import torch
import pickle
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils.data import from_numpy,to_numpy,ref_copy,collate_fn
from utils.torch_utils import torch_distributed_zero_first
from utils.general import check_path,ROOT
from scipy import sparse

from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
from skimage.transform import rotate


class PreprocessDataset(Dataset):
    def __init__(self, data_type ,config, flag=False):
        self.data_type= data_type if data_type!="test" else "test_obs"
        self.raw_path = os.path.join(config["raw_path"],self.data_type,"data")
        self.save_path = check_path(os.path.join(config["save_path"],self.data_type))
        assert os.path.exists(self.raw_path)
        self.avl = ArgoverseForecastingLoader(self.raw_path)
        self.avl.seq_list = sorted(self.avl.seq_list)
        self.am = ArgoverseMap()
        self.config = config
        self.length = len(self.avl)

    def __getitem__(self, idx):
        data = self.read_argo_data(idx)
        data = self.get_obj_feats(data)
        data['idx'] = idx
        data['city'] = self.avl[idx].city
        
        graph = self.get_lane_graph(data)
                
        new_graph = dict()
        new_graph['num_nodes'] = graph['num_nodes']
        new_graph['ctrs'] = graph['ctrs']
        new_graph['feats'] = graph['feats']
        new_graph['turn'] = graph['turn']
        new_graph['control'] = graph['control']
        new_graph['intersect'] = graph['intersect']
        new_graph['pre'] = graph['pre']
        new_graph['suc'] = graph['suc']
        new_graph['left'] = graph['left']
        new_graph['right'] = graph['right']

        new_data = dict()
        new_data['city'] = data['city']
        new_data['orig'] = data['orig']
        new_data['feats'] = data['feats']
        new_data['ctrs'] = data['ctrs']
        new_data['theta'] = data['theta']
        new_data['gt_preds'] = data['gt_preds']
        new_data['has_preds'] = data['has_preds']
        new_data['graph'] = new_graph

        return new_data
        
    def __len__(self):
        return self.length

    def read_argo_data(self, idx):
        df = self.avl[idx].seq_df
        """TIMESTAMP,TRACK_ID,OBJECT_TYPE,X,Y,CITY_NAME"""
        
        agt_ts = np.sort(np.unique(df['TIMESTAMP'].values))
        mapping = dict()
        for i, ts in enumerate(agt_ts):
            mapping[ts] = i

        trajs = np.concatenate((
            df.X.to_numpy().reshape(-1, 1),
            df.Y.to_numpy().reshape(-1, 1)), 1)
        
        steps = [mapping[x] for x in df['TIMESTAMP'].values]
        steps = np.asarray(steps, np.int32)

        objs = df.groupby(['TRACK_ID', 'OBJECT_TYPE']).groups
        keys = list(objs.keys())
        obj_type = [x[1] for x in keys]

        agt_idx = obj_type.index('AGENT')
        idcs = objs[keys[agt_idx]]
       
        agt_traj = trajs[idcs]
        agt_step = steps[idcs]

        del keys[agt_idx]
        ctx_trajs, ctx_steps = [], []
        for key in keys:
            idcs = objs[key]
            ctx_trajs.append(trajs[idcs])
            ctx_steps.append(steps[idcs])

        data = dict()
        data['trajs'] = [agt_traj] + ctx_trajs
        data['steps'] = [agt_step] + ctx_steps
        return data
    
    def get_obj_feats(self, data):
        orig = data['trajs'][0][19].copy().astype(float)

        if self.data_type=="train" and self.config['rot_aug']:
            theta = np.random.rand() * np.pi * 2.0
        else:
            pre = data['trajs'][0][18] - orig
            theta = np.pi - np.arctan2(pre[1], pre[0])

        rot = np.asarray([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]], float)

        feats, ctrs, gt_preds, has_preds = [], [], [], []
        for traj, step in zip(data['trajs'], data['steps']):
            if 19 not in step:
                continue

            gt_pred = np.zeros((30, 2), float)
            has_pred = np.zeros(30, bool)
            future_mask = np.logical_and(step >= 20, step < 50)
            post_step = step[future_mask] - 20
            post_traj = traj[future_mask]
            gt_pred[post_step] = post_traj
            has_pred[post_step] = 1
            
            obs_mask = step < 20
            step = step[obs_mask]
            traj = traj[obs_mask]
            idcs = step.argsort()
            step = step[idcs]
            traj = traj[idcs]
            
            for i in range(len(step)):
                if step[i] == 19 - (len(step) - 1) + i:
                    break
            step = step[i:]
            traj = traj[i:]

            feat = np.zeros((20, 3), float)
            feat[step, :2] = np.matmul(rot, (traj - orig.reshape(-1, 2)).T).T
            feat[step, 2] = 1.0

            x_min, x_max, y_min, y_max = self.config['pred_range']
            if feat[-1, 0] < x_min or feat[-1, 0] > x_max or feat[-1, 1] < y_min or feat[-1, 1] > y_max:
                continue

            ctrs.append(feat[-1, :2].copy())
            feat[1:, :2] -= feat[:-1, :2]
            feat[step[0], :2] = 0
            feats.append(feat)
            gt_preds.append(gt_pred)
            has_preds.append(has_pred)

        feats = np.asarray(feats, float)
        ctrs = np.asarray(ctrs, float)
        gt_preds = np.asarray(gt_preds, float)
        has_preds = np.asarray(has_preds, bool)

        data['feats'] = feats
        data['ctrs'] = ctrs
        data['orig'] = orig
        data['theta'] = theta
        data['rot'] = rot
        data['gt_preds'] = gt_preds
        data['has_preds'] = has_preds
        return data

    def get_lane_graph(self, data):
        """Get a rectangle area defined by pred_range."""
        x_min, x_max, y_min, y_max = self.config['pred_range']
        radius = max(abs(x_min), abs(x_max)) + max(abs(y_min), abs(y_max))
        lane_ids = self.am.get_lane_ids_in_xy_bbox(data['orig'][0], data['orig'][1], data['city'], radius)
        lane_ids = copy.deepcopy(lane_ids)
        
        lanes = dict()
        for lane_id in lane_ids:
            lane = self.am.city_lane_centerlines_dict[data['city']][lane_id]
            lane = copy.deepcopy(lane)
            centerline = np.matmul(data['rot'], (lane.centerline - data['orig'].reshape(-1, 2)).T).T
            x, y = centerline[:, 0], centerline[:, 1]
            if x.max() < x_min or x.min() > x_max or y.max() < y_min or y.min() > y_max:
                continue
            else:
                """Getting polygons requires original centerline"""
                polygon = self.am.get_lane_segment_polygon(lane_id, data['city'])
                polygon = copy.deepcopy(polygon)
                lane.centerline = centerline
                lane.polygon = np.matmul(data['rot'], (polygon[:, :2] - data['orig'].reshape(-1, 2)).T).T
                lanes[lane_id] = lane
            
        lane_ids = list(lanes.keys())
        ctrs, feats, turn, control, intersect = [], [], [], [], []
        for lane_id in lane_ids:
            lane = lanes[lane_id]
            ctrln = lane.centerline
            num_segs = len(ctrln) - 1
            
            ctrs.append(np.asarray((ctrln[:-1] + ctrln[1:]) / 2.0, float))
            feats.append(np.asarray(ctrln[1:] - ctrln[:-1], float))
            
            x = np.zeros((num_segs, 2), float)
            if lane.turn_direction == 'LEFT':
                x[:, 0] = 1
            elif lane.turn_direction == 'RIGHT':
                x[:, 1] = 1
            else:
                pass
            turn.append(x)

            control.append(lane.has_traffic_control * np.ones(num_segs, float))
            intersect.append(lane.is_intersection * np.ones(num_segs, float))
            
        node_idcs = []
        count = 0
        for i, ctr in enumerate(ctrs):
            node_idcs.append(range(count, count + len(ctr)))
            count += len(ctr)
        num_nodes = count
        
        pre, suc = dict(), dict()
        for key in ['u', 'v']:
            pre[key], suc[key] = [], []
        for i, lane_id in enumerate(lane_ids):
            lane = lanes[lane_id]
            idcs = node_idcs[i]
            
            pre['u'] += idcs[1:]
            pre['v'] += idcs[:-1]
            if lane.predecessors is not None:
                for nbr_id in lane.predecessors:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        pre['u'].append(idcs[0])
                        pre['v'].append(node_idcs[j][-1])
                    
            suc['u'] += idcs[:-1]
            suc['v'] += idcs[1:]
            if lane.successors is not None:
                for nbr_id in lane.successors:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        suc['u'].append(idcs[-1])
                        suc['v'].append(node_idcs[j][0])

        lane_idcs = []
        for i, idcs in enumerate(node_idcs):
            lane_idcs.append(i * np.ones(len(idcs), np.int64))
        lane_idcs = np.concatenate(lane_idcs, 0)

        pre_pairs, suc_pairs, left_pairs, right_pairs = [], [], [], []
        for i, lane_id in enumerate(lane_ids):
            lane = lanes[lane_id]

            nbr_ids = lane.predecessors
            if nbr_ids is not None:
                for nbr_id in nbr_ids:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        pre_pairs.append([i, j])

            nbr_ids = lane.successors
            if nbr_ids is not None:
                for nbr_id in nbr_ids:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        suc_pairs.append([i, j])

            nbr_id = lane.l_neighbor_id
            if nbr_id is not None:
                if nbr_id in lane_ids:
                    j = lane_ids.index(nbr_id)
                    left_pairs.append([i, j])

            nbr_id = lane.r_neighbor_id
            if nbr_id is not None:
                if nbr_id in lane_ids:
                    j = lane_ids.index(nbr_id)
                    right_pairs.append([i, j])
        pre_pairs = np.asarray(pre_pairs, np.int16)
        suc_pairs = np.asarray(suc_pairs, np.int16)
        left_pairs = np.asarray(left_pairs, np.int16)
        right_pairs = np.asarray(right_pairs, np.int16)
                    
        graph = dict()
        graph['ctrs'] = np.concatenate(ctrs, 0)
        graph['num_nodes'] = num_nodes
        graph['feats'] = np.concatenate(feats, 0)
        graph['turn'] = np.concatenate(turn, 0)
        graph['control'] = np.concatenate(control, 0)
        graph['intersect'] = np.concatenate(intersect, 0)
        graph['pre'] = [pre]
        graph['suc'] = [suc]
        graph['lane_idcs'] = lane_idcs
        graph['pre_pairs'] = pre_pairs
        graph['suc_pairs'] = suc_pairs
        graph['left_pairs'] = left_pairs
        graph['right_pairs'] = right_pairs
        
        for k1 in ['pre', 'suc']:
            for k2 in ['u', 'v']:
                graph[k1][0][k2] = np.asarray(graph[k1][0][k2], np.int32)
        
        for key in ['pre', 'suc']:
            graph[key] += self.dilated_nbrs(graph[key][0], graph['num_nodes'], self.config['num_scales'])
        
        # out = self.get_left_right_ori(to_long(from_numpy(graph)), self.config['cross_dist'])
        out = self.get_left_right(graph, self.config['cross_dist'])
        graph["left"] =out["left"]
        graph["right"]=out["right"]

        return graph

    def get_left_right(self,graph,cross_dist, cross_angle=None):
        """get_left_rigth : 修改左右邻居关系,主要是扩增
        """
        left, right = dict(), dict()
        lane_idcs = graph['lane_idcs']
        num_nodes = len(lane_idcs)
        num_lanes = lane_idcs[-1].item() + 1

        dist = np.expand_dims(graph['ctrs'],axis=1) - np.expand_dims(graph['ctrs'],axis=0)
        dist = np.sqrt((dist ** 2).sum(2))
        hi = np.arange(num_nodes).reshape(-1, 1).repeat(num_nodes,axis=1).reshape(-1)
        wi = np.arange(num_nodes).reshape(1, -1).repeat(num_nodes, axis=0).reshape(-1)
        row_idcs = np.arange(num_nodes)

        if cross_angle is not None:
            f1 = graph['feats'][hi]
            f2 = graph['ctrs'][wi] - graph['ctrs'][hi]
            t1 = np.arctan2(f1[:, 1], f1[:, 0])
            t2 = np.arctan2(f2[:, 1], f2[:, 0])
            dt = t2 - t1
            m = dt > 2 * np.pi
            dt[m] = dt[m] - 2 * np.pi
            m = dt < -2 * np.pi
            dt[m] = dt[m] + 2 * np.pi

            mask = np.logical_and(dt > 0, dt < eval(self.config['cross_angle']))
            left_mask = np.logical_not(mask)
            mask = np.logical_and(dt < 0, dt > -eval(self.config['cross_angle']))
            right_mask = np.logical_not(mask)

        pre = np.zeros((num_lanes, num_lanes),dtype=float)
        suc = np.zeros((num_lanes, num_lanes),dtype=float)
        pre[graph['pre_pairs'][:, 0], graph['pre_pairs'][:, 1]] = 1
        suc[graph['suc_pairs'][:, 0], graph['suc_pairs'][:, 1]] = 1

        pairs = graph['left_pairs']
        if len(pairs) > 0:
            mat = np.zeros((num_lanes, num_lanes),dtype=float)
            mat[pairs[:, 0], pairs[:, 1]] = 1
            mat = (np.matmul(mat, pre) + np.matmul(mat, suc) + mat) > 0.5

            left_dist = dist.copy()
            mask = np.logical_not(mat[lane_idcs[hi], lane_idcs[wi]])
            left_dist[hi[mask], wi[mask]] = 1e6
            if cross_angle is not None:
                left_dist[hi[left_mask], wi[left_mask]] = 1e6

            min_dist = left_dist.min(1)
            min_idcs = left_dist.argmin(1)
            mask = min_dist < cross_dist
            ui = row_idcs[mask]
            vi = min_idcs[mask]
            f1 = graph['feats'][ui]
            f2 = graph['feats'][vi]
            t1 = np.arctan2(f1[:, 1], f1[:, 0])
            t2 = np.arctan2(f2[:, 1], f2[:, 0])
            dt = np.abs(t1 - t2)
            m = dt > np.pi
            dt[m] = np.abs(dt[m] - 2 * np.pi)
            m = dt < 0.25 * np.pi

            ui = ui[m]
            vi = vi[m]

            left['u'] = ui.astype(np.int16)
            left['v'] = vi.astype(np.int16)
        else:
            left['u'] = np.zeros(0, np.int16)
            left['v'] = np.zeros(0, np.int16)

        pairs = graph['right_pairs']
        if len(pairs) > 0:
            mat = np.zeros((num_lanes, num_lanes),dtype=float)
            mat[pairs[:, 0], pairs[:, 1]] = 1
            mat = (np.matmul(mat, pre) + np.matmul(mat, suc) + mat) > 0.5

            right_dist = dist.copy()
            mask = np.logical_not(mat[lane_idcs[hi], lane_idcs[wi]])
            right_dist[hi[mask], wi[mask]] = 1e6
            if cross_angle is not None:
                right_dist[hi[right_mask], wi[right_mask]] = 1e6

            min_dist = right_dist.min(1)
            min_idcs = right_dist.argmin(1)
            mask = min_dist < cross_dist
            ui = row_idcs[mask]
            vi = min_idcs[mask]
            f1 = graph['feats'][ui]
            f2 = graph['feats'][vi]
            t1 = np.arctan2(f1[:, 1], f1[:, 0])
            t2 = np.arctan2(f2[:, 1], f2[:, 0])
            dt = np.abs(t1 - t2)
            m = dt > np.pi
            dt[m] = np.abs(dt[m] - 2 * np.pi)
            m = dt < 0.25 * np.pi

            ui = ui[m]
            vi = vi[m]

            right['u'] = ui.astype(np.int16)
            right['v'] = vi.astype(np.int16)
        else:
            right['u'] = np.zeros(0, np.int16)
            right['v'] = np.zeros(0, np.int16)

        out = dict()
        out['left'] = left
        out['right'] = right
        return out

    def get_left_right_ori(self,graph,cross_dist,cross_angle=None):
        left, right = dict(), dict()

        lane_idcs = graph['lane_idcs']
        num_nodes = len(lane_idcs)
        num_lanes = lane_idcs[-1].item() + 1

        dist = graph['ctrs'].unsqueeze(1) - graph['ctrs'].unsqueeze(0)
        dist = torch.sqrt((dist ** 2).sum(2))
        hi = torch.arange(num_nodes).long().to(dist.device).view(-1, 1).repeat(1, num_nodes).view(-1)
        wi = torch.arange(num_nodes).long().to(dist.device).view(1, -1).repeat(num_nodes, 1).view(-1)
        row_idcs = torch.arange(num_nodes).long().to(dist.device)

        if cross_angle is not None:
            f1 = graph['feats'][hi]
            f2 = graph['ctrs'][wi] - graph['ctrs'][hi]
            t1 = torch.atan2(f1[:, 1], f1[:, 0])
            t2 = torch.atan2(f2[:, 1], f2[:, 0])
            dt = t2 - t1
            m = dt > 2 * np.pi
            dt[m] = dt[m] - 2 * np.pi
            m = dt < -2 * np.pi
            dt[m] = dt[m] + 2 * np.pi
            mask = torch.logical_and(dt > 0, dt < self.config['cross_angle'])
            left_mask = mask.logical_not()
            mask = torch.logical_and(dt < 0, dt > -self.config['cross_angle'])
            right_mask = mask.logical_not()

        pre = graph['pre_pairs'].new().float().resize_(num_lanes, num_lanes).zero_()
        pre[graph['pre_pairs'][:, 0], graph['pre_pairs'][:, 1]] = 1
        suc = graph['suc_pairs'].new().float().resize_(num_lanes, num_lanes).zero_()
        suc[graph['suc_pairs'][:, 0], graph['suc_pairs'][:, 1]] = 1

        pairs = graph['left_pairs']
        if len(pairs) > 0:
            mat = pairs.new().float().resize_(num_lanes, num_lanes).zero_()
            mat[pairs[:, 0], pairs[:, 1]] = 1
            mat = (torch.matmul(mat, pre) + torch.matmul(mat, suc) + mat) > 0.5

            left_dist = dist.clone()
            mask = mat[lane_idcs[hi], lane_idcs[wi]].logical_not()
            left_dist[hi[mask], wi[mask]] = 1e6
            if cross_angle is not None:
                left_dist[hi[left_mask], wi[left_mask]] = 1e6

            min_dist, min_idcs = left_dist.min(1)
            mask = min_dist < cross_dist
            ui = row_idcs[mask]
            vi = min_idcs[mask]
            f1 = graph['feats'][ui]
            f2 = graph['feats'][vi]
            t1 = torch.atan2(f1[:, 1], f1[:, 0])
            t2 = torch.atan2(f2[:, 1], f2[:, 0])
            dt = torch.abs(t1 - t2)
            m = dt > np.pi
            dt[m] = torch.abs(dt[m] - 2 * np.pi)
            m = dt < 0.25 * np.pi

            ui = ui[m]
            vi = vi[m]

            left['u'] = ui.cpu().numpy().astype(np.int16)
            left['v'] = vi.cpu().numpy().astype(np.int16)
        else:
            left['u'] = np.zeros(0, np.int16)
            left['v'] = np.zeros(0, np.int16)

        pairs = graph['right_pairs']
        if len(pairs) > 0:
            mat = pairs.new().float().resize_(num_lanes, num_lanes).zero_()
            mat[pairs[:, 0], pairs[:, 1]] = 1
            mat = (torch.matmul(mat, pre) + torch.matmul(mat, suc) + mat) > 0.5

            right_dist = dist.clone()
            mask = mat[lane_idcs[hi], lane_idcs[wi]].logical_not()
            right_dist[hi[mask], wi[mask]] = 1e6
            if cross_angle is not None:
                right_dist[hi[right_mask], wi[right_mask]] = 1e6

            min_dist, min_idcs = right_dist.min(1)
            mask = min_dist < cross_dist
            ui = row_idcs[mask]
            vi = min_idcs[mask]
            f1 = graph['feats'][ui]
            f2 = graph['feats'][vi]
            t1 = torch.atan2(f1[:, 1], f1[:, 0])
            t2 = torch.atan2(f2[:, 1], f2[:, 0])
            dt = torch.abs(t1 - t2)
            m = dt > np.pi
            dt[m] = torch.abs(dt[m] - 2 * np.pi)
            m = dt < 0.25 * np.pi

            ui = ui[m]
            vi = vi[m]

            right['u'] = ui.cpu().numpy().astype(np.int16)
            right['v'] = vi.cpu().numpy().astype(np.int16)
        else:
            right['u'] = np.zeros(0, np.int16)
            right['v'] = np.zeros(0, np.int16)

        out = dict()
        out['left'] = left
        out['right'] = right
        # out['idx'] = graph['idx']
        return out

    def dilated_nbrs(self,nbr, num_nodes, num_scales):
        data = np.ones(len(nbr['u']), bool)
        csr = sparse.csr_matrix((data, (nbr['u'], nbr['v'])), shape=(num_nodes, num_nodes))

        mat = csr
        nbrs = []
        for i in range(1, num_scales):
            mat = mat * mat

            nbr = dict()
            coo = mat.tocoo()
            nbr['u'] = coo.row.astype(np.int64)
            nbr['v'] = coo.col.astype(np.int64)
            nbrs.append(nbr)
        return nbrs

class ArgoDataset(Dataset):
    def __init__(self, data_type, config):
        self.data_type= data_type if data_type!="test" else "test_obs"
        self.raw_path = os.path.join(config["raw_path"],self.data_type)
        self.save_path = os.path.join(config["save_path"],self.data_type)
        assert os.path.exists(self.save_path)
        self.avl = ArgoverseForecastingLoader(self.raw_path)
        self.avl.seq_list = sorted(self.avl.seq_list)
        self.am = ArgoverseMap()
        self.config = config
        self.length = 0
        for x in Path(self.save_path).iterdir():
            if x.is_file() and x.suffix == ".pkl":
                self.length+=1
        # super(ArgoDataset,self).__init__(data_type=data_type,config=config)
        
    def __getitem__(self, idx):
        file_path = os.path.join(self.save_path,f"{idx}.pkl")
        with open(file_path,'rb') as f:
            data = pickle.load(f)
        
        rot = np.asarray([
            [np.cos(data["theta"]), -np.sin(data["theta"])],
            [np.sin(data["theta"]), np.cos(data["theta"])]], float)
        data["rot"] = rot

        if self.data_type == "test_obs":
            print(self.avl)
            data['argo_id'] = int(self.avl.seq_list[idx].name[:-4])
            data['city'] = self.avl[idx].city
        
        # if self.data_type == "train" and self.config['rot_aug']:
        #     new_data = dict()
        #     for key in ['city', 'orig', 'gt_preds', 'has_preds']:
        #         if key in data:
        #             new_data[key] = ref_copy(data[key])

        #     dt = np.random.rand() * self.config['rot_size']#np.pi * 2.0
        #     theta = data['theta'] + dt
        #     new_data['theta'] = theta
        #     new_data['rot'] = np.asarray([
        #         [np.cos(theta), -np.sin(theta)],
        #         [np.sin(theta), np.cos(theta)]], float)

        #     rot = np.asarray([
        #         [np.cos(-dt), -np.sin(-dt)],
        #         [np.sin(-dt), np.cos(-dt)]], float)
        #     new_data['feats'] = data['feats'].copy()
        #     new_data['feats'][:, :, :2] = np.matmul(new_data['feats'][:, :, :2], rot)
        #     new_data['ctrs'] = np.matmul(data['ctrs'], rot)

        #     graph = dict()
        #     for key in ['num_nodes', 'turn', 'control', 'intersect', 'pre', 'suc', 'lane_idcs', 'left_pairs', 'right_pairs', 'left', 'right']:
        #         graph[key] = ref_copy(data['graph'][key])
        #     graph['ctrs'] = np.matmul(data['graph']['ctrs'], rot)
        #     graph['feats'] = np.matmul(data['graph']['feats'], rot)
        #     new_data['graph'] = graph
        #     data = new_data
        # else:
        #     new_data = dict()
        #     for key in ['city', 'orig', 'gt_preds', 'has_preds', 'theta', 'rot', 'feats', 'ctrs', 'graph']:
        #         if key in data:
        #             new_data[key] = ref_copy(data[key])
        #     data = new_data
        
        # if 'raster' in self.config and self.config['raster']:
        #     data.pop('graph')
        #     x_min, x_max, y_min, y_max = self.config['pred_range']
        #     cx, cy = data['orig']
        #     region = [cx + x_min, cx + x_max, cy + y_min, cy + y_max]
        #     raster = self.map_query.query(region, data['theta'], data['city'])
        #     data['raster'] = raster

        return data

    def __len__(self):
        return self.length


def preprocess(args):
    """ use raw data to generate preprocess data """

    with open(args.config,"r",encoding="utf-8") as f:
        config = yaml.load(f,Loader=yaml.FullLoader)

    dataset = PreprocessDataset(
        data_type = args.data_type,
        config=config
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        collate_fn=from_numpy,
        pin_memory=False,
        drop_last=False,
    )
    for i in tqdm(range(dataset.__len__())):
        if args.start_iter>0:
            if i < args.start_iter:
                continue
        data = dataset.__getitem__(i)
        with open(f"{dataset.save_path}/{i}.pkl","wb") as f:
                pickle.dump(data,f)

    # for i,data in enumerate(tqdm(loader)):
    #     if args.start_iter>0:
    #         if (i+1) * args.batch_size < args.start_iter:
    #             continue
    #     for j in range(len(data)):
    #         with open(f"{dataset.save_path}/{i*len(data)+j}.pkl","wb") as f:
    #             pickle.dump(to_numpy(data[j]),f)

def create_dataloader(data_type,batch_size,workers,rank,shuffle,config):
    # with torch_distributed_zero_first(rank):
    #     dataset = ArgoDataset(data_type,config)
    dataset = ArgoDataset(data_type,config)
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = DistributedSampler(dataset,shuffle=shuffle) if rank != -1 else None
    dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=nw,
                sampler=sampler,
                pin_memory=True,
                collate_fn= collate_fn
    )
    return dataloader, dataset

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Data preprocess for argo forcasting dataset")
    parser.add_argument("--data_type",type=str, default="train")  # train val or test
    parser.add_argument("--config",type=str, default=f"{str(ROOT)}/config/data.yaml") # config file
    parser.add_argument("--workers",type=int, default=0)
    parser.add_argument("--batch_size",type=int, default=16)
    parser.add_argument("--start_iter",type=int, default=0)
    args = parser.parse_args()
    
    preprocess(args)
    


