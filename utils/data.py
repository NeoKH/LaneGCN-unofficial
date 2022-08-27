import h5py
import torch
import numpy as np

def write_h5py(data,save_path):
    with h5py.File(save_path, "w") as f:
        # f.create_dataset('idx',         data = data["idx"])
        # f.create_dataset('city',        data = data["city"])
        f.create_dataset('feats',       data = data["feats"])
        f.create_dataset('ctrs',        data = data["ctrs"])
        f.create_dataset('orig',        data = data["orig"])
        f.create_dataset('theta',       data = data["theta"])
        f.create_dataset('rot',         data = data["rot"])
        f.create_dataset('gt_preds',    data = data["gt_preds"])
        f.create_dataset('has_preds',   data = data["has_preds"])
        _graph = f.create_group("graph")
        _graph.create_dataset('ctrs',     data = data["graph"]["ctrs"])
        _graph.create_dataset('num_nodes',data = data["graph"]["num_nodes"])
        _graph.create_dataset('feats',    data = data["graph"]["feats"])
        _graph.create_dataset('turn',     data = data["graph"]["turn"])
        _graph.create_dataset('control',  data = data["graph"]["control"])
        _graph.create_dataset('intersect',data = data["graph"]["intersect"])
        # _graph.create_dataset('pre_pairs',data = data["graph"]["pre_pairs"])
        # _graph.create_dataset('suc_pairs',data = data["graph"]["suc_pairs"])
        # _graph.create_dataset('lfet_pairs',data = data["graph"]["left_pairs"])
        # _graph.create_dataset('right_pairs',data = data["graph"]["right_pairs"])
        pre = _graph.create_group("pre")
        for k in range(len(data["graph"]["pre"])):
            tmp = pre.create_group(str(k))
            tmp.create_dataset("u",data = data["graph"]["pre"][k]["u"])
            tmp.create_dataset("v",data = data["graph"]["pre"][k]["v"])
        suc = _graph.create_group("suc")
        for k in range(len(data["graph"]["suc"])):
            tmp = suc.create_group(str(k))
            tmp.create_dataset("u",data = data["graph"]["suc"][k]["u"])
            tmp.create_dataset("v",data = data["graph"]["suc"][k]["v"])
        left = _graph.create_group("left")
        left.create_dataset('u',      data = data["graph"]["left"]["u"])
        left.create_dataset('v',      data = data["graph"]["left"]["v"])
        right = _graph.create_group("right")
        right.create_dataset('u',      data = data["graph"]["right"]["u"])
        right.create_dataset('v',      data = data["graph"]["right"]["v"])

def read_h5py(file_path):
    data=dict()
    with h5py.File(file_path,"r") as f:
        for key0 in f.keys():
            if key0 != "graph":
                data[key0] = f[key0][()]
            else:
                data["graph"]=dict()
                for key1 in f["graph"].keys():
                    if key1 not in ["pre","suc","left","right"]:
                        data["graph"][key1] = f["graph"][key1][()]
                    elif key1 in ["pre","suc"]:
                        tmp_list = []
                        tmp_dict = dict()
                        for key2 in sorted(f["graph"][key1].keys()):
                            for key3 in f["graph"][key1][key2]:
                                tmp_dict[key3] = f["graph"][key1][key2][key3]
                            tmp_list.append(tmp_dict)
                        data["graph"][key1] = tmp_list    
                    elif key1 in ["left","right"]:
                        for key2 in f["graph"][key1].keys():
                            data["graph"][key1][key2] = f["graph"][key1][key2][()]   

def ref_copy(data):
    if isinstance(data, list):
        return [ref_copy(x) for x in data]
    if isinstance(data, dict):
        d = dict()
        for key in data:
            d[key] = ref_copy(data[key])
        return d
    return data

def collate_fn(batch):
    batch = from_numpy(batch)
    batch = to_float32(batch)
    batch = to_long(batch)
    return_batch = dict()
    # Batching by use a list for non-fixed size
    for key in batch[0].keys():
        return_batch[key] = [x[key] for x in batch]
    return return_batch
    
def from_numpy(data):
    """Recursively transform numpy.ndarray to torch.Tensor.
    """
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = from_numpy(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [from_numpy(x) for x in data]
    if isinstance(data, np.ndarray):
        """Pytorch now has bool type."""
        data = torch.from_numpy(data)
    return data

def cat(batch):
    if torch.is_tensor(batch[0]):
        batch = [x.unsqueeze(0) for x in batch]
        return_batch = torch.cat(batch, 0)
    elif isinstance(batch[0], list) or isinstance(batch[0], tuple):
        batch = zip(*batch)
        return_batch = [cat(x) for x in batch]
    elif isinstance(batch[0], dict):
        return_batch = dict()
        for key in batch[0].keys():
            return_batch[key] = cat([x[key] for x in batch])
    else:
        return_batch = batch
    return return_batch

def to_numpy(data):
    """Recursively transform torch.Tensor to numpy.ndarray.
    """
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = to_numpy(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [to_numpy(x) for x in data]
    if torch.is_tensor(data):
        data = data.numpy()
    return data

def to_float32(data):
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = to_float32(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [to_float32(x) for x in data]
    if isinstance(data, torch.Tensor) and data.dtype in [torch.float64,torch.float16,torch.double]:
        data = data.type(torch.float32)
    return data

def to_int16(data):
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = to_int16(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [to_int16(x) for x in data]
    if isinstance(data, np.ndarray) and data.dtype==np.int64:
        data = data.astype(np.int16)
    return data

def to_long(data):
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = to_long(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [to_long(x) for x in data]
    if torch.is_tensor(data) and data.dtype in [torch.int16,torch.int32]:
        data = data.long()
    return data

