import importlib.machinery
import os.path as osp

import torch

from .fps import sample, sample_idx

torch.ops.load_library(
    importlib.machinery.PathFinder().find_spec(f"_core", [osp.dirname(__file__)]).origin
)

__all__ = ["sample", "sample_idx"]


