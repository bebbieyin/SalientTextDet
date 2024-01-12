import torch


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