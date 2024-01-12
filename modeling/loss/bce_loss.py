"""
BCE with OHEM
OHEM controlled by negative ratio
"""

import torch
from torch import nn
import torch.nn.functional as F


def bce_ohem(logits, targets, ohem_ratio=0.3):
    # Flatten logits and targets to 1D tensors
    logits_flat = logits.view(-1)
    targets_flat = targets.view(-1)

    # Calculate binary cross-entropy loss
    #  nn.BCEWithLogitsLoss apply sigmoid() internally 
    #  add it manually if you are using nn.BCELoss.
    #bce_loss = F.binary_cross_entropy_with_logits(logits_flat, targets_flat, reduction='none')
    #bce_loss = F.binary_cross_entropy(logits_flat, targets_flat, reduction='none')

    loss = nn.BCEWithLogitsLoss(reduction='none')
    bce_loss = loss(logits_flat, targets_flat)

    # Calculate the number of hard negative examples to select
    num_examples = bce_loss.size(0)
    num_selected = int(ohem_ratio * num_examples)

    # Sort the loss values in descending order
    sorted_loss, indices = torch.sort(bce_loss, descending=True)

    # Select the top `num_selected` hard negatives
    selected_loss = sorted_loss[:num_selected]

    # Compute the mean loss over the selected hard negatives
    loss = torch.mean(selected_loss)


    return loss

def bce_loss_noOHEM(logits, targets, ohem_ratio=0.2):
    # Flatten logits and targets to 1D tensors
    logits_flat = logits.view(-1)
    targets_flat = targets.view(-1)

    # Calculate binary cross-entropy loss
    #  nn.BCEWithLogitsLoss apply sigmoid() internally 
    #  add it manually if you are using nn.BCELoss.
    #bce_loss = F.binary_cross_entropy_with_logits(logits_flat, targets_flat, reduction='none')
    bce_loss = F.binary_cross_entropy_with_logits(logits_flat, targets_flat, reduction='mean')


    return bce_loss
