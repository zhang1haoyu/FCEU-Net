# -*- coding:utf-8  -*-

import torch
from collections import OrderedDict
from torch.nn.functional import cross_entropy, binary_cross_entropy_with_logits
from torch import Tensor
from torch.nn import functional as F
from lib.criterion.bce_dice import dice_bce_loss
from collections import OrderedDict
import torch.nn as nn
from lib.criterion.soft_ce import SoftCrossEntropyLoss
from lib.criterion.joint_loss import JointLoss
from lib.criterion.dice import DiceLoss

def CriterionSet(loss='cross_entropy'):
    if loss == 'cross_entropy':
        return CrossEntropyLoss()
    elif loss == 'bce_dice_loss':
        return BCE_Dice_Loss()
    elif loss == 'bce_logits_loss':
        return BCEWithLogitsLoss()
    elif loss == 'EdgeLoss':
        return EdgeLoss()
    elif loss == 'cross_entropy_single':
        return CrossEntropyLoss_single()







class BCEWithLogitsLoss:
    def __call__(self, inputs, target):
        losses = {}
        '''for name, x in inputs.items():'''
        for name, x in inputs.items():
            xh, xw = x.size(2), x.size(3)
            h, w = target.size(2), target.size(3)

            if xh != h or xw != w:
                x = F.interpolate(
                    input=x, size=(h, w),
                    mode='bilinear', align_corners=True
                )

            losses[name] = binary_cross_entropy_with_logits(x, target)

        if len(losses) == 1:
            return losses['out']

        return losses['out'] + 0.5 * losses['aux']


class CrossEntropyLoss:
    def __call__(self, inputs, target):
        losses = {}
        '''for name, x in inputs.items():'''
        for name, x in inputs.items():
            xh, xw = x.size(2), x.size(3)
            h, w = target.size(1), target.size(2)

            if xh != h or xw != w:
                x = F.interpolate(
                    input=x, size=(h, w),
                    mode='bilinear', align_corners=True
                )

            losses[name] = cross_entropy(x, target)

        if len(losses) == 1:
            return losses['out']

        return losses['out'] + 0.5 * losses['aux']

class CrossEntropyLoss_single:
    def __call__(self, inputs, target):
        xh, xw = inputs.size(2), inputs.size(3)
        h, w = target.size(1), target.size(2)

        if xh != h or xw != w:
            inputs = F.interpolate(
                input=inputs, size=(h, w),
                mode='bilinear', align_corners=True
            )

        loss = cross_entropy(inputs, target)
        return loss


class BCE_Dice_Loss:
    def __init__(self):
        self.loss_fn = dice_bce_loss()

    def __call__(self, inputs, target):
        losses = {}
        '''for name, x in inputs.items():'''
        for name, x in inputs.items():
            xh, xw = x.size(2), x.size(3)
            h, w = target.size(2), target.size(3)

            if xh != h or xw != w:
                x = F.interpolate(
                    input=x, size=(h, w),
                    mode='bilinear', align_corners=True
                )

            losses[name] = self.loss_fn(x, target)

        if len(losses) == 1:
            return losses['out']

        return losses['out'] + 0.5 * losses['aux']


'''import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeLoss(nn.Module):
    def __init__(self, ignore_index=255, edge_factor=1.0, class_weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(EdgeLoss, self).__init__()
        self.main_loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                                   DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
        self.edge_factor = edge_factor
        self.class_weights = torch.tensor(class_weights) if class_weights else None  # Assuming class_weights is a torch.tensor

    def get_boundary(self, x):
        # Your implementation for boundary extraction
        if x is None:
            raise ValueError("Input tensor x is None.")
        device = x.device
        laplacian_kernel_target = torch.tensor(
            [-1, -1, -1, -1, 8, -1, -1, -1, -1],
            dtype=torch.float32).reshape(1, 1, 3, 3).to(device)
        x = x.unsqueeze(1).float()
        x = F.conv2d(x, laplacian_kernel_target, padding=1)
        x = x.clamp(min=0)
        x[x >= 0.1] = 1
        x[x < 0.1] = 0

        return x


    def compute_edge_loss(self, logits, targets):
        bs = logits.size()[0]
        boundary_targets = self.get_boundary(targets)
        boundary_targets = boundary_targets.view(bs, 1, -1)
        logits = F.softmax(logits, dim=1).argmax(dim=1).squeeze(dim=1)
        boundary_pre = self.get_boundary(logits)
        boundary_pre = boundary_pre / (boundary_pre + 0.01)
        boundary_pre = boundary_pre.view(bs, 1, -1)
        edge_loss = F.binary_cross_entropy_with_logits(boundary_pre, boundary_targets)

        return edge_loss

    def forward(self, logits, targets):
        if isinstance(logits, OrderedDict):
            logits = logits['out']

        # Compute main loss
        main_loss = self.main_loss(logits, targets)

        # Compute edge loss
        edge_loss = self.compute_edge_loss(logits, targets)

        # Apply class weights
        if self.class_weights is not None:
            class_weights = self.class_weights.to(logits.device)
            main_loss = main_loss * class_weights
            edge_loss = edge_loss * class_weights

        # Calculate total loss across all samples
        total_main_loss = main_loss.mean()  # Or sum() depending on your preference
        total_edge_loss = edge_loss.mean()  # Or sum()

        # Combine losses
        loss = (total_main_loss + total_edge_loss * self.edge_factor) / (self.edge_factor + 1)

        return loss'''


class EdgeLoss(nn.Module):
    def __init__(self, ignore_index=255, edge_factor=1.0):
        super(EdgeLoss, self).__init__()
        self.main_loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                                   DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
        self.edge_factor = edge_factor

    def get_boundary(self, x):
        device = x.device
        laplacian_kernel_target = torch.tensor(
            [-1, -1, -1, -1, 8, -1, -1, -1, -1],
            dtype=torch.float32).reshape(1, 1, 3, 3).to(device)
        x = x.unsqueeze(1).float()
        x = F.conv2d(x, laplacian_kernel_target, padding=1)
        x = x.clamp(min=0)
        x[x >= 0.1] = 1
        x[x < 0.1] = 0

        return x

    def compute_edge_loss(self, logits, targets):
        bs = logits.size()[0]
        boundary_targets = self.get_boundary(targets)
        boundary_targets = boundary_targets.view(bs, 1, -1)
        logits = F.softmax(logits, dim=1).argmax(dim=1).squeeze(dim=1)
        boundary_pre = self.get_boundary(logits)
        boundary_pre = boundary_pre / (boundary_pre + 0.01)
        boundary_pre = boundary_pre.view(bs, 1, -1)
        edge_loss = F.binary_cross_entropy_with_logits(boundary_pre, boundary_targets)

        return edge_loss

    def compute_kl_loss(self, p, q, pad_mask=None):
        p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
        q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

        # pad_mask is for seq-level tasks
        if pad_mask is not None:
            p_loss.masked_fill_(pad_mask, 0.)
            q_loss.masked_fill_(pad_mask, 0.)

        # You can choose whether to use function "sum" and "mean" depending on your task
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()

        loss = (p_loss + q_loss) / 2
        return loss

    '''def forward(self, logits, targets):
        if isinstance(logits, OrderedDict):
            logits = logits['out']
        loss = (self.main_loss(logits, targets) + self.compute_edge_loss(logits, targets) * self.edge_factor) / (self.edge_factor + 1)

        return loss
'''
    def forward(self, logits, logits2, targets):
        if isinstance(logits, OrderedDict):
            logits = logits['out']
        if isinstance(logits2, OrderedDict):
            logits2 = logits2['out']

        main_loss1 = self.main_loss(logits, targets)
        edge_loss1 = self.compute_edge_loss(logits, targets)
        ce_loss1 = (main_loss1 + edge_loss1 * self.edge_factor) / (self.edge_factor + 1)

        main_loss2 = self.main_loss(logits2, targets)
        edge_loss2 = self.compute_edge_loss(logits2, targets)
        ce_loss2 = (main_loss2 + edge_loss2 * self.edge_factor) / (self.edge_factor + 1)

        ce_loss = 0.5 * (ce_loss1 + ce_loss2)
        kl_loss = self.compute_kl_loss(logits, logits2)

        loss = ce_loss + 0.5 * kl_loss
        return loss


class OHEM_CELoss(nn.Module):

    def __init__(self, thresh=0.7, ignore_index=255):
        super(OHEM_CELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).cuda()
        self.ignore_index = ignore_index
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, logits, labels):
        n_min = labels[labels != self.ignore_index].numel() // 16
        loss = self.criteria(logits, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)


class UnetFormerLoss(nn.Module):

    def __init__(self, ignore_index=255):
        super().__init__()
        self.main_loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                                   DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
        self.aux_loss = SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index)

    def forward(self, logits, labels):
        if self.training and len(logits) == 2:
            logit_main, logit_aux = logits
            loss = self.main_loss(logit_main, labels) + 0.4 * self.aux_loss(logit_aux, labels)
        else:
            loss = self.main_loss(logits, labels)

        return loss



from torchvision import transforms
import numpy as np
from PIL import Image

if __name__ == '__main__':
    torch.manual_seed(0)
    predict = OrderedDict()
    predict['out'] = torch.randn((1, 3, 3, 3), dtype=torch.float32)
    mask = torch.zeros((1, 3, 3), dtype=torch.long)
    loss = EdgeLoss()

    print(loss(predict, mask))

