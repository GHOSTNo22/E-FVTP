import time
from path import Path

_CURRENT_DIR = Path(__file__).parent.abspath()
import sys

sys.path.append(_CURRENT_DIR)
sys.path.append(_CURRENT_DIR.parent)
import json
import os
import pickle
import random
from argparse import ArgumentParser
import pandas as pd

import numpy as np
import torch
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, EMNIST, MNIST, FashionMNIST

from constants import MEAN, STD
from partition import dirichlet_distribution, randomly_assign_classes, randomly_assign_AIS
from utils.dataset import CIFARDataset, MNISTDataset, AISDataset, dataset_to_Dataset

from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
lengths = 4
targets = 1
DATASET = {
    "mnist": (MNIST, MNISTDataset),
    "emnist": (EMNIST, MNISTDataset),
    "fmnist": (FashionMNIST, MNISTDataset),
    "cifar10": (CIFAR10, CIFARDataset),
    "cifar100": (CIFAR100, CIFARDataset),
    #"AIS": (AIS, AISDataset),
}


def main(args):
    def data_start():
        T = 1000
        x = torch.arange(1, T + 1, dtype=torch.float32)
        ais_data = pd.read_csv('data/AIS/datafull.csv',encoding='unicode_escape')
        y = ais_data.iloc[:args.client_num_in_total*1000, [0, 1, 2, 4, 10]]
        #len_y = int(len(ais_data)/1000)
        #print(len_y)
        #sample_nums = random.sample(range(len_y), args.client_num_in_total)
        #print(sample_nums)
        #y = None
        #for i in range(args.client_num_in_total):
        #    if i == 0:
        #        y = ais_data.iloc[sample_nums[i]*1000:sample_nums[i]*1000+1000,[0, 1, 2, 4, 10]]
        #    else:
        #        y = pd.concat([y, ais_data.iloc[sample_nums[i]*1000:sample_nums[i]*1000+1000,[0, 1, 2, 4, 10]]])
         #       print(sample_nums[i])
         #       print(ais_data.iloc[sample_nums[i]*1000:sample_nums[i]*1000+1000,[0, 1, 2, 4, 10]])
           # print(y)
        #print(y)
        min_max_scaler = preprocessing.MinMaxScaler()
        y_minmax = min_max_scaler.fit_transform(y)
        # print(y.head())
        y = pd.DataFrame(y_minmax)
        # print(y.head())
        return x, y

    def data_prediction_to_f_and_t(data, num_features, num_targets):

        features, target = [], []
        F,T = [], []
        judge = 0
        for i in range(((len(data) - num_features - num_targets) // num_targets) + 1):
            f = data.loc[i * num_targets:i * num_targets + num_features - 1, :]
            t = data.loc[i * num_targets + num_features:i * num_targets + num_features + num_targets - 1, :]
            if f.loc[i, 4].item() == t[4].item():
                features.append(f.iloc[:, [0, 1, 2, 3]])
                target.append(t.iloc[:, [0, 1, 2, 3]])
           #     judge = 0
           # elif f.loc[i,4].item() != t[4].item() and judge == 0:
           #     judge = 1
           #     F.append(features)
           #     T.append(target)
           #     features = []
           #     target = []
        # print(features[98])
        # print(target[98])
        return np.array(features), np.array(target)

    def dataset_split_4sets(data_features, data_target, ratio=0.8):
        split_index = int(ratio * len(data_features))
        train_features = data_features[:split_index]
        train_target = data_target[:split_index]
        test_features = data_features[split_index:]
        test_target = data_target[split_index:]
        return train_features, train_target, test_features, test_target

    _DATASET_ROOT = (
        Path(args.root).abspath() / args.dataset
        if args.root is not None
        else _CURRENT_DIR.parent / args.dataset
    )
    if args.dataset != 'AIS':
        _PICKLES_DIR = _CURRENT_DIR.parent / args.dataset / "pickles"

        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

        classes_map = None
        transform = transforms.Compose(
            [
                transforms.Normalize(MEAN[args.dataset], STD[args.dataset]),
            ]
        )
        target_transform = None

        if not os.path.isdir(_DATASET_ROOT):
            os.makedirs(_DATASET_ROOT)
        if os.path.isdir(_PICKLES_DIR):
            os.system("rm -rf {_PICKLES_DIR}".format(_PICKLES_DIR = _PICKLES_DIR))
        os.system("mkdir -p {_PICKLES_DIR}".format(_PICKLES_DIR = _PICKLES_DIR))

        client_num_in_total = args.client_num_in_total
        client_num_in_total = args.client_num_in_total
        ori_dataset, target_dataset = DATASET[args.dataset]
        if args.dataset == "emnist":
            trainset = ori_dataset(
                _DATASET_ROOT,
                train=True,
                download=True,
                split=args.emnist_split,
                transform=transforms.ToTensor(),
            )
            testset = ori_dataset(
                _DATASET_ROOT,
                train=False,
                split=args.emnist_split,
                transform=transforms.ToTensor(),
            )
        else:
            trainset = ori_dataset(
                _DATASET_ROOT,
                train=True,
                download=True,
            )
            testset = ori_dataset(
                _DATASET_ROOT,
                train=False,
            )
        concat_datasets = [trainset, testset]
        if args.alpha > 0:  # NOTE: Dirichlet(alpha)
            all_datasets, stats = dirichlet_distribution(
                ori_dataset=concat_datasets,
                target_dataset=target_dataset,
                num_clients=client_num_in_total,
                alpha=args.alpha,
                transform=transform,
                target_transform=target_transform,
            )
        else:  # NOTE: sort and partition
            classes = len(ori_dataset.classes) if args.classes <= 0 else args.classes
            all_datasets, stats = randomly_assign_classes(
                ori_datasets=concat_datasets,
                target_dataset=target_dataset,
                num_clients=client_num_in_total,
                num_classes=classes,
                transform=transform,
                target_transform=target_transform,
            )

        for subset_id, client_id in enumerate(
            range(0, len(all_datasets), args.client_num_in_each_pickles)
        ):
            subset = []
            for dataset in all_datasets[
                client_id : client_id + args.client_num_in_each_pickles
            ]:
                #print("len(dataset)")
                #print(len(dataset))
                num_val_samples = int(len(dataset) * args.valset_ratio)
                num_test_samples = int(len(dataset) * args.test_ratio)
                num_train_samples = len(dataset) - num_val_samples - num_test_samples
                #print(len(num_val_samples))
                #print(len(num_test_samples))
                #print(len(num_train_samples))
                train, val, test = random_split(
                    dataset, [num_train_samples, num_val_samples, num_test_samples]
                )
                subset.append({"train": train, "val": val, "test": test})
            with open(_PICKLES_DIR / str(subset_id) + ".pkl", "wb") as f:
                pickle.dump(subset, f)

        # save stats
        if args.type == "user":
            train_clients_num = int(client_num_in_total * args.fraction)
            clients_4_train = [i for i in range(train_clients_num)]
            clients_4_test = [i for i in range(train_clients_num, client_num_in_total)]

            with open(_PICKLES_DIR / "seperation.pkl", "wb") as f:
                pickle.dump(
                    {
                        "train": clients_4_train,
                        "test": clients_4_test,
                        "total": client_num_in_total,
                    },
                    f,
                )

            train_clients_stats = dict(
                zip(clients_4_train, list(stats.values())[:train_clients_num])
            )
            test_clients_stats = dict(
                zip(
                    clients_4_test,
                    list(stats.values())[train_clients_num:],
                )
            )

            with open(_CURRENT_DIR.parent / args.dataset / "all_stats.json", "w") as f:
                json.dump({"train": train_clients_stats, "test": test_clients_stats}, f)

        else:  # NOTE: "sample"  save stats
            client_id_indices = [i for i in range(client_num_in_total)]
            with open(_PICKLES_DIR / "seperation.pkl", "wb") as f:
                pickle.dump(
                    {
                        "id": client_id_indices,
                        "total": client_num_in_total,
                    },
                    f,
                )
            with open(_CURRENT_DIR.parent / args.dataset / "all_stats.json", "w") as f:
                json.dump(stats, f)

        args.root = (
            Path(args.root).abspath()
            if str(_DATASET_ROOT) != str(_CURRENT_DIR.parent / args.dataset)
            else None
        )


    # todo dataset
    else :
        _PICKLES_DIR = _CURRENT_DIR.parent / args.dataset / "pickles"

        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

        classes_map = None

        if not os.path.isdir(_DATASET_ROOT):
            os.makedirs(_DATASET_ROOT)
        if os.path.isdir(_PICKLES_DIR):
            os.system("rm -rf {_PICKLES_DIR}".format(_PICKLES_DIR = _PICKLES_DIR))
        os.system("mkdir -p {_PICKLES_DIR}".format(_PICKLES_DIR = _PICKLES_DIR))

        client_num_in_total = args.client_num_in_total
        _, data = data_start()
        data_pd = []
        datasets = []
        f,t = data_prediction_to_f_and_t(data, lengths, targets)
        #print(np.shape(f))
        #print(np.shape(t))
        datasets = dataset_to_Dataset(data_features=f, data_target=t)
        #print(np.shape(datasets))
        #print(type(datasets))
       # for i in range(len(f)):
       #     datasets.append(dataset_to_Dataset(data_features=f[i], data_target=t[i]))
#        concat_datasets = [trainset, testset]
        if args.alpha > 0:  # NOTE: Dirichlet(alpha)
            all_datasets, stats = dirichlet_distribution(
                ori_dataset=concat_datasets,
                target_dataset=target_dataset,
                num_clients=client_num_in_total,
                alpha=args.alpha,
                # transform=transform,
                # target_transform=target_transform,
            )
        else:  # NOTE: sort and partition
            # classes = len(ori_dataset.classes) if args.classes <= 0 else args.classes
            all_datasets, stats = randomly_assign_AIS(
                ori_datasets=datasets,
                target_dataset=dataset_to_Dataset,
                num_clients=client_num_in_total,
                rate_clients=args.rate_clients
            )
            
        for subset_id, client_id in enumerate(
                range(0, len(all_datasets), args.client_num_in_each_pickles)
        ):
            subset = []
            for dataset in all_datasets[client_id : client_id+client_num_in_total]:
                #print(len(dataset))
                num_val_samples = int(len(dataset) * args.valset_ratio)
                num_test_samples = int(len(dataset) * args.test_ratio)
                num_train_samples = len(dataset) - num_val_samples - num_test_samples
                train, val, test = random_split(
                    dataset, [num_train_samples, num_val_samples, num_test_samples]
                )
                #print(num_val_samples)
                #print(num_test_samples)
                #print(num_train_samples)
                subset.append({"train": train, "val": val, "test": test})
            with open(_PICKLES_DIR / str(subset_id) + ".pkl", "wb") as f:
                pickle.dump(subset, f)

        # save stats
        if args.type == "user":
            train_clients_num = int(client_num_in_total * args.fraction)
            clients_4_train = [i for i in range(train_clients_num)]
            clients_4_test = [i for i in range(train_clients_num, client_num_in_total)]

            with open(_PICKLES_DIR / "seperation.pkl", "wb") as f:
                pickle.dump(
                    {
                        "train": clients_4_train,
                        "test": clients_4_test,
                        "total": client_num_in_total,
                    },
                    f,
                )

            train_clients_stats = dict(
                zip(clients_4_train, list(stats.values())[:train_clients_num])
            )
            test_clients_stats = dict(
                zip(
                    clients_4_test,
                    list(stats.values())[train_clients_num:],
                )
            )

            with open(_CURRENT_DIR.parent / args.dataset / "all_stats.json", "w") as f:
                json.dump({"train": train_clients_stats, "test": test_clients_stats}, f)

        else:  # NOTE: "sample"  save stats
            client_id_indices = [i for i in range(client_num_in_total)]
            with open(_PICKLES_DIR / "seperation.pkl", "wb") as f:
                pickle.dump(
                    {
                        "id": client_id_indices,
                        "total": client_num_in_total,
                    },
                    f,
                )
            with open(_CURRENT_DIR.parent / args.dataset / "all_stats.json", "w") as f:
                json.dump(stats, f)

        args.root = (
            Path(args.root).abspath()
            if str(_DATASET_ROOT) != str(_CURRENT_DIR.parent / args.dataset)
            else None
        )

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=[
            "mnist",
            "cifar10",
            "cifar100",
            "emnist",
            "fmnist",
            "AIS"
        ],
        default="mnist",
    )

    parser.add_argument("--client_num_in_total", type=int, default=10)
    parser.add_argument(
        "--fraction", type=float, default=0.9, help="Propotion of train clients"
    )
    parser.add_argument("--valset_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument(
        "--classes",
        type=int,
        default=-1,
        help="Num of classes that one client's data belong to.",
    )
    parser.add_argument("--seed", type=int, default=int(time.time()))
    ################# Dirichlet distribution only #################
    parser.add_argument(
        "--alpha",
        type=float,
        default=0,
        help="Only for controling data hetero degree while performing Dirichlet partition.",
    )
    ###############################################################

    ################# For EMNIST only #####################
    parser.add_argument(
        "--emnist_split",
        type=str,
        choices=["byclass", "bymerge", "letters", "balanced", "digits", "mnist"],
        default="byclass",
    )
    #######################################################
    parser.add_argument(
        "--type", type=str, choices=["sample", "user"], default="sample"
    )
    parser.add_argument("--client_num_in_each_pickles", type=int, default=10)
    # parser.add_argument("--root", type=str, default="/root/repos/python/mine/datasets")
    parser.add_argument("--rate_clients", type=float, default=0.1)
    parser.add_argument("--root", type=str, default="/mnt/VMSTORE/fedves/root")
    args = parser.parse_args()
    main(args)
    args_dict = dict(args._get_kwargs())
    with open(_CURRENT_DIR.parent / "args.json", "w") as f:
        json.dump(args_dict, f)



