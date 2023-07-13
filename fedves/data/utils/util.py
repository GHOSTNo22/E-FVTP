import json
import math
import os
import pickle
from typing import Dict, List, Tuple, Union

from path import Path
from torch.utils.data import Subset, random_split

_CURRENT_DIR = Path(__file__).parent.abspath()
_ARGS_DICT = json.load(open(_CURRENT_DIR.parent / "args.json", "r"))
lengths = 4
targets = 1


def get_dataset(
    dataset: str,
    client_id: int,
) -> Dict[str, Subset]:
    # todo edited

    client_num_in_each_pickles = _ARGS_DICT["client_num_in_each_pickles"]
    pickles_dir = _CURRENT_DIR.parent / dataset / "pickles"
    if os.path.isdir(pickles_dir) is False:
        raise RuntimeError("Please preprocess and create pickles first.")

    pickle_path = (
        pickles_dir / f"{math.floor(client_id / client_num_in_each_pickles)}.pkl"
    )
    with open(pickle_path, "rb") as f:
        subset = pickle.load(f)
    client_dataset = subset[client_id % client_num_in_each_pickles]
    trainset = client_dataset["train"]
    valset = client_dataset["val"]
    testset = client_dataset["test"]

    # client_num_in_each_csv = _ARGS_DICT["client_num_in_each_pickles"]
    # csv_dir = _CURRENT_DIR.parent / dataset / "data.csv"
    # if os.path.isdir(csv_dir) is False:
    #     raise RuntimeError("Please preprocess and create data.scv first.")
    # x, y = data_start()
    # dataset_features, dataset_target = data_prediction_to_f_and_t(y, lengths, targets)
    # train_features, train_target, test_features, test_target = dataset_split_4sets(dataset_features, dataset_target)
    # train_features, train_target, val_features, val_target = dataset_split_4sets(train_features, train_target)
    #
    # trainset = dataset_to_Dataset(data_features=train_features, data_target=train_target)
    # valset = dataset_to_Dataset(data_features=val_features, data_target=val_target)
    # testset = dataset_to_Dataset(data_features=test_features, data_target=test_target)

    return {"train": trainset, "val": valset, "test": testset}


def get_client_id_indices(
    dataset,
) -> Union[Tuple[List[int], List[int], int], Tuple[List[int], int]]:
    pickles_dir = _CURRENT_DIR.parent / dataset / "pickles"
    with open(pickles_dir / "seperation.pkl", "rb") as f:
        seperation = pickle.load(f)
    if _ARGS_DICT["type"] == "user":
        return seperation["train"], seperation["test"], seperation["total"]
    else:  # NOTE: "sample"
        return seperation["id"], seperation["total"]

