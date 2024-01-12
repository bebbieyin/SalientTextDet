import torch
from torch import nn



# class NormAffine(nn.Module):
#     def __init__(self):
#         super(NormAffine, self).__init__()

#     def forward(self, mat, eps=1e-7, method='sum'):
#         matdim = len(mat.size())
#         if method == 'sum':
#             tempsum = torch.sum(mat, dim=(matdim - 1, matdim - 2), keepdim=True) + eps
#             out = mat / tempsum
#         elif method == 'one':
#             (tempmin, _) = torch.min(mat, dim=matdim - 1, keepdim=True)
#             (tempmin, _) = torch.min(tempmin, dim=matdim - 2, keepdim=True)
#             tempmat = mat - tempmin
#             (tempmax, _) = torch.max(tempmat, dim=matdim - 1, keepdim=True)
#             (tempmax, _) = torch.max(tempmax, dim=matdim - 2, keepdim=True)
#             tempmax = tempmax + eps
#             out = tempmat / tempmax
#         else:
#             raise NotImplementedError('Map method [%s] is not implemented' % method)
#         return out


# class KL_loss(nn.Module):
#     def __init__(self):
#         super(KL_loss, self).__init__()

#         self.norm = NormAffine()
    

#     def forward(self, pred, gt):
#         assert pred.size() == gt.size()
#         pred = self.norm(pred, eps=1e-7, method='sum')
#         gt = self.norm(gt, eps=1e-7, method='sum')
#         loss = torch.sum(gt * torch.log(1e-7 + gt / (pred + 1e-7)))
#         loss = loss / pred.size()[0]
#         return loss

def NormAffine(mat, eps=1e-7,
               method='sum'):  # tensor [batch_size, channels, image_height, image_width] normalize each fea map;
    matdim = len(mat.size())
    if method == 'sum':
        tempsum = torch.sum(mat, dim=(matdim - 1, matdim - 2), keepdim=True) + eps
        out = mat / tempsum
    elif method == 'one':
        (tempmin, _) = torch.min(mat, dim=matdim - 1, keepdim=True)
        (tempmin, _) = torch.min(tempmin, dim=matdim - 2, keepdim=True)
        tempmat = mat - tempmin
        (tempmax, _) = torch.max(tempmat, dim=matdim - 1, keepdim=True)
        (tempmax, _) = torch.max(tempmax, dim=matdim - 2, keepdim=True)
        tempmax = tempmax + eps
        out = tempmat / tempmax
    else:
        raise NotImplementedError('Map method [%s] is not implemented' % method)
    return out


def KL_loss(out, gt):
    assert out.size() == gt.size()
    out = NormAffine(out, eps=1e-7, method='sum')
    gt = NormAffine(gt, eps=1e-7, method='sum')
    loss = torch.sum(gt * torch.log(1e-7 + gt / (out + 1e-7)))
    loss = loss / out.size()[0]
    return loss

"""
a = torch.randn([1,3,896,896])
b = torch.randn([1,3,896,896])
saliency_loss = KL_loss(a,a)

print(saliency_loss)
"""