# -*- coding: utf-8 -*-
# @Time    : 10/1/21
# @Author  : GXYM
import torch
import torch.nn as nn
from cfglib.config import config as cfg
from network.Seg_loss import SegmentLoss
from network.Reg_loss import PolyMatchingLoss
import torch.nn.functional as F


class TextLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.MSE_loss = torch.nn.MSELoss(reduce=False, size_average=False)
        self.BCE_loss = torch.nn.BCELoss(reduce=False, size_average=False)
        self.PolyMatchingLoss = PolyMatchingLoss(cfg.num_points, cfg.device)
        self.KL_loss = torch.nn.KLDivLoss(reduce=False, size_average=False)

    @staticmethod
    def single_image_loss(pre_loss, loss_label):
        batch_size = pre_loss.shape[0]
        sum_loss = torch.mean(pre_loss.view(-1)) * 0
        pre_loss = pre_loss.view(batch_size, -1)
        loss_label = loss_label.view(batch_size, -1)
        eps = 0.001
        for i in range(batch_size):
            average_number = 0
            positive_pixel = len(pre_loss[i][(loss_label[i] >= eps)])
            average_number += positive_pixel
            if positive_pixel != 0:
                posi_loss = torch.mean(pre_loss[i][(loss_label[i] >= eps)])
                sum_loss += posi_loss
                if len(pre_loss[i][(loss_label[i] < eps)]) < 3 * positive_pixel:
                    nega_loss = torch.mean(pre_loss[i][(loss_label[i] < eps)])
                    average_number += len(pre_loss[i][(loss_label[i] < eps)])
                else:
                    nega_loss = torch.mean(torch.topk(pre_loss[i][(loss_label[i] < eps)], 3 * positive_pixel)[0])
                    average_number += 3 * positive_pixel
                sum_loss += nega_loss
            else:
                nega_loss = torch.mean(torch.topk(pre_loss[i], 100)[0])
                average_number += 100
                sum_loss += nega_loss
            # sum_loss += loss/average_number

        return sum_loss/batch_size

    def cls_ohem(self, predict, target, train_mask, negative_ratio=3.):
        pos = (target * train_mask).bool()
        neg = ((1 - target) * train_mask).bool()

        n_pos = pos.float().sum()
        if n_pos.item() > 0:
            loss_pos = self.BCE_loss(predict[pos], target[pos]).sum()
            loss_neg = self.BCE_loss(predict[neg], target[neg])
            n_neg = min(int(neg.float().sum().item()), int(negative_ratio * n_pos.float()))
        else:
            loss_pos = torch.tensor(0.)
            loss_neg = self.BCE_loss(predict[neg], target[neg])
            n_neg = 100
        loss_neg, _ = torch.topk(loss_neg, n_neg)

        return (loss_pos + loss_neg.sum()) / (n_pos + n_neg).float()

    @staticmethod
    def loss_calc_flux(pred_flux, gt_flux, weight_matrix, mask, train_mask):

        # norm loss
        gt_flux = 0.999999 * gt_flux / (gt_flux.norm(p=2, dim=1).unsqueeze(1) + 1e-3)
        norm_loss = weight_matrix * torch.mean((pred_flux - gt_flux) ** 2, dim=1)*train_mask
        norm_loss = norm_loss.sum(-1).mean()
        # norm_loss = norm_loss.sum()

        # angle loss
        mask = train_mask * mask
        pred_flux = 0.999999 * pred_flux / (pred_flux.norm(p=2, dim=1).unsqueeze(1) + 1e-3)
        # angle_loss = weight_matrix * (torch.acos(torch.sum(pred_flux * gt_flux, dim=1))) ** 2
        # angle_loss = angle_loss.sum(-1).mean()
        angle_loss = (1 - torch.cosine_similarity(pred_flux, gt_flux, dim=1))
        angle_loss = angle_loss[mask].mean()

        return norm_loss, angle_loss

    @staticmethod
    def get_poly_energy(energy_field, img_poly, ind, h, w):
        img_poly = img_poly.clone().float()
        img_poly[..., 0] = img_poly[..., 0] / (w / 2.) - 1
        img_poly[..., 1] = img_poly[..., 1] / (h / 2.) - 1

        batch_size = energy_field.size(0)
        gcn_feature = torch.zeros([img_poly.size(0), energy_field.size(1), img_poly.size(1)]).to(img_poly.device)
        for i in range(batch_size):
            poly = img_poly[ind == i].unsqueeze(0)
            gcn_feature[ind == i] = torch.nn.functional.grid_sample(energy_field[i:i + 1], poly)[0].permute(1, 0, 2)
        return gcn_feature

    def loss_energy_regularization(self, energy_field, img_poly, inds, h, w):
        energys = []
        for i, py in enumerate(img_poly):
            energy = self.get_poly_energy(energy_field.unsqueeze(1), py, inds, h, w)
            energys.append(energy.squeeze(1).sum(-1))

        regular_loss = torch.tensor(0.)
        energy_loss = torch.tensor(0.)
        for i, e in enumerate(energys[1:]):
            regular_loss += torch.clamp(e - energys[i], min=0.0).mean()
            energy_loss += torch.where(e <= 0.01, torch.tensor(0.), e).mean()

        return (energy_loss+regular_loss)/len(energys[1:])

    def forward(self, pred, train_mask, eps=None):
        """
          calculate boundary proposal network loss
        """

        # if cfg.scale > 1:
        #     train_mask = F.interpolate(train_mask.float().unsqueeze(1),
        #                                scale_factor=1/cfg.scale, mode='bilinear').squeeze().bool()
        #     tr_mask = F.interpolate(tr_mask.float().unsqueeze(1),
        #                             scale_factor=1/cfg.scale, mode='bilinear').squeeze().bool()
        # pixel class loss
        cls_loss = self.cls_ohem(pred, tr_mask.float(), train_mask)
        #cls_loss = self.BCE_loss(pred,  train_mask.float())
        #cls_loss = torch.mul(cls_loss, train_mask.float()).mean()


        if eps is None:
            alpha = 1.0; beta = 3.0; theta=0.5; gama = 0.05
        else:
            alpha = 1.0; beta = 3.0; theta=0.5;
            gama = 0.1*torch.sigmoid(torch.tensor((eps - cfg.max_epoch)/cfg.max_epoch))
        loss = alpha*cls_loss 

      

        return cls_loss

