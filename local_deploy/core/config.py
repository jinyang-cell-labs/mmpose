# Copyright (c) OpenMMLab. All rights reserved.
import torch
import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def resolve_device(device):
    if device in (None, 'auto'):
        return 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return device
