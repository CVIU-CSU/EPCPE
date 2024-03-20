# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import MODELS
from .utils import weighted_loss
import torch


@weighted_loss
def rot_loss(pred: Tensor, target: Tensor) -> Tensor:
    """A Wrapper of ROT loss.
    Args:
        pred (Tensor): The prediction.
        target (Tensor): The learning target of the prediction.

    Returns:
        Tensor: loss Tensor
    """
    eps = 1e-6
    pred=pred.reshape(-1,3,3)
    target=target.reshape(-1,3,3)
    product = torch.bmm(pred, target.transpose(1, 2))
    trace = torch.sum(product[:, torch.eye(3).bool()], 1)
    theta = torch.clamp(0.5 * (trace - 1), -1 + eps, 1 - eps)
    rad = torch.acos(theta)

    return rad.reshape(-1,1)


@MODELS.register_module()
class ROTLoss(nn.Module):
    """ROTLoss.

    Args:
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
    """

    def __init__(self,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function of loss.

        Args:
            pred (Tensor): The prediction.
            target (Tensor): The learning target of the prediction.
            weight (Tensor, optional): Weight of the loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        Returns:
            Tensor: The calculated loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * rot_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss

