# AbsRel.py
# TorchMetrics implementation of Abs Rel metric for depth maps.

from types import NoneType
import torch
import torch.nn as nn
import torchmetrics


class AbsRelRunningAvg(torchmetrics.Metric):
    """ Implements running average of Abs Rel metric. """
    higher_is_better = False
    full_state_update = True

    def __init__(self, args):
        super().__init__()
        self.add_state("batch_count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("running_avg", default=torch.tensor(0), dist_reduce_fx="mean")

    
    def update(self, depth_pred: torch.Tensor, depth_gt: torch.Tensor):
        """Expects pre-masked depth_pred and depth_gt (i.e. only valid values.)"""
        val = torch.mean(torch.abs(depth_gt - depth_pred) / depth_gt)
        
        self.running_avg = (val + (self.running_avg * self.batch_count)) / (self.batch_count + 1)
        self.batch_count += 1


    def compute(self):
        return self.running_avg



class AbsRel(torchmetrics.Metric):
    """Implements the Abs Rel metric for depth maps.
    """
    higher_is_better = False
    full_state_update = False


    def __init__(self, args):
        super().__init__()
        self.add_state("normed_abs_diff_total", default=torch.Tensor([0]), dist_reduce_fx="sum")
        self.add_state("valid_pixel_count", default=torch.Tensor([0]), dist_reduce_fx="sum")

    
    def update(self, depth_pred: torch.Tensor, depth_gt: torch.Tensor):
        """Expects pre-masked depth_pred and depth_gt (i.e. only valid values.)"""
        self.normed_abs_diff_total += torch.sum(torch.abs(depth_gt - depth_pred) / depth_gt)
        self.valid_pixel_count += torch.numel(depth_gt)
    

    def compute(self):
        return (self.normed_abs_diff_total / self.valid_pixel_count).float()
