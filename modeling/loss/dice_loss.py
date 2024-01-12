import torch
from torch import nn
import torch.nn.functional as F

def dice_loss(pred, target, smooth=1.):
    """Dice loss
    """
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()




# a = torch.sigmoid(torch.randn([4,1,224,224]))
# b = torch.sigmoid(torch.randn([4,1,224,224]))

# loss2 = dice_loss(a,b)
# print(loss2)