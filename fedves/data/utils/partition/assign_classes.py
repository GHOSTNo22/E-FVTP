import random
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from torch.utils.data import Dataset


def sort_and_alloc(
    datasets: List[Dataset], num_clients: int, num_classes: int
) -> Dict[int, np.ndarray]:
    total_sample_nums = sum(map(lambda ds: len(ds), datasets))
    #classes == 1? 非分类工作不需要分类切片
    num_shards = num_clients
    # one shard's length indicate how many data samples that belongs to one class that one client can obtain.
    size_of_shards = int(total_sample_nums / num_shards)

    dict_users = {i: np.array([], dtype=np.int64) for i in range(num_clients)}

    # labels = np.concatenate([ds.targets for ds in datasets], axis=0, dtype=np.int64)
    labels = np.concatenate([ds.target for ds in datasets], axis=0, dtype=np.int64)
    idxs = np.arange(total_sample_nums)

    # sort sample indices according to labels
    idxs_labels = np.vstack((idxs, labels))
    # corresponding labels after sorting are [0, .., 0, 1, ..., 1, ...]
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # assign
    idx_shard = [i for i in range(num_shards)]
    for i in range(num_clients):
        rand_set = random.sample(idx_shard, num_classes)
        idx_shard = list(set(idx_shard) - set(rand_set))
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (
                    dict_users[i],
                    idxs[rand * size_of_shards : (rand + 1) * size_of_shards],
                ),
                axis=0,
            )

    return dict_users


def randomly_assign_classes(
    ori_datasets: List[Dataset],
    target_dataset: Dataset,
    num_clients: int,
    num_classes: int,
    transform=None,
    target_transform=None,
) -> Tuple[List[Dataset], Dict[str, Dict[str, int]]]:
    stats = {}
    dict_users = sort_and_alloc(ori_datasets, num_clients, num_classes)
    targets_numpy = np.concatenate(
        [ds.targets for ds in ori_datasets], axis=0, dtype=np.int64
    )
    data_numpy = np.concatenate(
        [ds.data for ds in ori_datasets], axis=0, dtype=np.float32
    )
    datasets = []
    for i, indices in dict_users.items():
        stats[i] = {"x": None, "y": None}
        stats[i]["x"] = len(indices)
        stats[i]["y"] = Counter(targets_numpy[indices].tolist())
        datasets.append(
            target_dataset(
                data=data_numpy[indices],
                targets=targets_numpy[indices],
                transform=transform,
                target_transform=target_transform,
            )
        )
    return datasets, stats

def sort_and_alloc_AIS(
    datasets: List[Dataset], num_clients: int, rate_clients: float
) -> Dict[int, np.ndarray]:
    #total_sample_nums = sum(map(lambda ds: len(ds), datasets))
    total_sample_nums = len(datasets)
    print("total_sample_nums:"+str(total_sample_nums))
    #classes == 1? 非分类工作不需要分类切片
    num_shards = num_clients
    # one shard's length indicate how many data samples that belongs to one class that one client can obtain.
    size_of_shards = int(total_sample_nums / num_shards)

    dict_users = {i: np.array([], dtype=np.int64) for i in range(num_clients)}

    # labels = np.concatenate([ds.targets for ds in datasets], axis=0, dtype=np.int64)
    #labels = np.concatenate([ds.target for ds in datasets], axis=0, dtype=np.float64)
    idxs = np.arange(total_sample_nums)
   # print("idxs.size:"+str(idxs))
   # print("labels.size:"+str(labels))

    # sort sample indices according to labels
   # idxs_labels = np.vstack((idxs, labels))
    # corresponding labels after sorting are [0, .., 0, 1, ..., 1, ...]
   # idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    #idxs = idxs_labels[0, :]
    print("size_of_shards:"+str(size_of_shards))
    # assign
    #print("len(datasets):",len(datasets))
    idx_shard = [i for i in range(num_shards)]
    for i in range(num_clients):
        #修改num_classes为idx_shard*rate_clients
        rand_set = random.sample(idx_shard, 1)
        #print(rand_set)
        idx_shard = list(set(idx_shard) - set(rand_set))
        #print(len(idx_shard))
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (
                    dict_users[i],
                    idxs[rand * size_of_shards : (rand + 1) * size_of_shards],
                ),
                axis=0,
            )
        #for rand in rand_set:
           # print("rand:",rand)
           # print(dict_users[i])
        #dict_users[i] = rand_set
    #print(dict_users)
    return dict_users


def randomly_assign_AIS(
    ori_datasets: List[Dataset],
    target_dataset: Dataset,
    num_clients: int,
    rate_clients: float,
) -> Tuple[List[Dataset], Dict[str, Dict[str, int]]]:
    stats = {}
    dict_users = sort_and_alloc_AIS(ori_datasets, num_clients, rate_clients)
    print(type(ori_datasets))
    print(np.shape(ori_datasets))
    #targets_numpy = np.concatenate(
    #    [target for target in ori_datasets.target], axis=0, dtype=np.float64
    #)
    #print(targets_numpy)
    #data_numpy = np.concatenate(
    #    [features for features in ori_datasets.features], axis=0, dtype=np.float32
    #)
   # print(data_numpy)
    targets_numpy = ori_datasets.target.numpy()
    data_numpy = ori_datasets.features.numpy()

    datasets = []
    subset = []
    #for i, indices in dict_users.items():
       # stats[i] = {"x": None, "y": None}
       # stats[i]["x"] = len(ori_datasets[indices])
       # stats[i]["y"] = len(ori_datasets[indices])
       # for rand in indices:
       #     for dataset in ori_datasets[rand]:
       #         subset.append(dataset)
                
       # datasets.append(
       #     subset
       # )
    #print(targets_numpy[1])
    #print(data_numpy[1])
    for i, indices in dict_users.items():
        stats[i] = {"x": None, "y": None}
        stats[i]["x"] = len(indices)
        stats[i]["y"] = len(targets_numpy[indices])
        #print(indices)
        datasets.append(
            target_dataset(
                data_features=data_numpy[indices],
                data_target=targets_numpy[indices],
            ) 
        )
    return datasets, stats
