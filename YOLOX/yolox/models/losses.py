#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from yolox.utils import cxcywh2xyxy


def cross_entropy_soft_targets(input, target, reduction='none'):
    log_input = torch.nn.functional.log_softmax(input, dim=-1)
    loss = -torch.sum(target * log_input, dim=-1)

    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise NotImplementedError('Unsupported reduction mode.')


class BoxUncertainLoss(nn.Module):
    def __init__(self, reduction='none', loss_type='nll'):
        super(BoxUncertainLoss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred_mean, pred_var, target, err_target, target_obj):
        if self.loss_type == 'nll_na':
            assert pred_mean.shape[0] == pred_var.shape[0] == target.shape[0]
            loss = (0.5 * pred_var + 0.5 * torch.exp(-pred_var) * torch.abs(pred_mean - target)).sum(1) / 4
        else:
            assert pred_mean.shape[0] == pred_var.shape[0] == target.shape[0] == err_target.shape[0] == target_obj.shape[0]
            max_len = target[:, 0].max().int()
            count, target = (target[:, 0] / 4).int(), target[:, 1:1+max_len]
            max_count = (max_len / 4).int()

            # repeat mean and var
            pred_mean = cxcywh2xyxy(pred_mean).repeat(1, max_count).masked_fill_(target == 0, 0)
            if self.loss_type == 'nll':
                pred_var = pred_var.repeat(1, max_count).masked_fill_(target == 0, 0)
                loss = (0.5 * pred_var + 0.5 * torch.exp(-pred_var) * torch.abs(pred_mean - target)).sum(1) / (count * 4)
            elif self.loss_type == 'dmm':
                pred_var = pred_var.repeat(1, max_count).masked_fill_(target == 0, 0)
                loss = (
                           torch.abs(pred_mean - target) + torch.abs(torch.exp(pred_var) - ((pred_mean - target) ** 2))
                       ).sum(1) / (count * 4)
            elif self.loss_type == 'dmm':
                alpha = 1 - target_obj
                # z = stats.norm.ppf(1 - alpha / 2)
                z = torch.distributions.Normal(0, 1).icdf(1 - torch.clamp(alpha, min=.01) / 2)

                # Calculate the standard deviation
                # assert torch.all(err_target >= 0)
                target_var = (err_target / z) ** 2
                loss = ((torch.abs(pred_mean - target)).sum(1) / (count * 4) +
                        torch.abs(torch.exp(pred_var) - target_var).masked_fill_(target_var == 0, 0).sum(1) / 4
                        )
            else:
                raise ValueError(f'Invalid loss type: {self.loss_type}')

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


if __name__ == '__main__':
    import time
    bu_loss = BoxUncertainLoss(reduction='mean', loss_type='dmm')
    pred_mean = torch.tensor([[10., 20, 10, 20], [400, 200, 50, 80]], requires_grad=True)
    pred_var = torch.tensor([[1., 0.5, -1, 0.4], [-3, 1.5, 2, -1]], requires_grad=True)
    target = torch.tensor([[4, 3, 10, 10, 20, 0, 0, 0, 0, 0, 0, 0, 0],
                           [12, 380, 175, 420, 225, 379, 180, 433, 212, 382, 173, 411, 230]])
    st = time.time()
    print(bu_loss(pred_mean.repeat(1000,1), pred_var.repeat(1000,1), target.repeat(1000,1)))
    print(time.time() - st)

class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss
