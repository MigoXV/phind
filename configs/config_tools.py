import os
import json
import torch
from omegaconf import OmegaConf


def load_config(config_path: str, config_set: str = None) -> OmegaConf:
    # if config_set is None:
    #     config_set = []
    config = OmegaConf.load(config_path)
    if config_set:
        if isinstance(config_set, str):
            config_set = config_set.strip("[]").split(",")
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(config_set))

    return config


def load_train_config(config_path: str, config_set: str = None) -> OmegaConf:
    config = load_config(config_path, config_set)

    # config = OmegaConf.structured(TrainConfig)
    if config.num_workers == "auto":
        config.num_workers = os.cpu_count()

    if config.device == "auto":
        config.device = "cuda" if torch.cuda.is_available() else "cpu"

    return config


def load_evaluate_config(
    config_path: str,
    data_dict_json_path: str,
    checkpoint_path: str,
    model_name: str,
    device: str,
    val_batch_size: int,
    num_workers: str,
) -> OmegaConf:
    config = OmegaConf.create()
    config.data_dict_json_path = data_dict_json_path
    config.checkpoint_path = checkpoint_path
    config.model_name = model_name
    config.device = device
    config.val_batch_size = val_batch_size
    config.num_workers = num_workers

    if config.num_workers == "auto":
        config.num_workers = os.cpu_count()

    if config.device == "auto":
        config.device = "cuda" if torch.cuda.is_available() else "cpu"

    if config_path:
        config_from_path = OmegaConf.load(config_path)
        config = OmegaConf.merge(config, config_from_path)

    return config
