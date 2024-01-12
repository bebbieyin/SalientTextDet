import sys
sys.path.append('/home/yinyin/salient_text')

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from detectron2.modeling import build_backbone
from detectron2.config import get_cfg
from modeling.low_level_model import LowLevelFeat
from modeling.swin_fpn.swin_transformer import * 
from modeling.swin_fpn.config import *
from modeling.saliency_detector import define_salD
from modeling.text_detector import TextDetector


class TextModel(nn.Module):
    def __init__(self,cfg,input_size=896, device='cuda'):
        super(TextModel, self).__init__()

        inner_channels = 1280
        self.input_size = input_size
        self.device = device

        """ Swin + FPN Backbone """
        self.backbone = build_backbone(cfg)
        
        self.upsample = nn.Upsample(size= None , mode='bilinear', align_corners=False)
        self.salhead = define_salD(inner_channels, netSalD='denseNet_15layer')
        self.low_salhead = define_salD(13, netSalD='denseNet_15layer')

        self.texthead = TextDetector(inner_channels)
        
        self.low_level = LowLevelFeat(img_size=self.input_size)

    def step_function(self, x, y):
            return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))
        
    def forward(self, x):
        """ Base network """
        fpn_feat = self.backbone(x)
        
        """ Upsample and fuse FPN features"""
        fpn_feat_list= list(fpn_feat.values())
        max_height = max([fm.size(2) for fm in fpn_feat_list])
        max_width = max([fm.size(3) for fm in fpn_feat_list])
        self.upsample.size = [max_height, max_width] 
        upsampled_feature_maps = [self.upsample(fm) for fm in fpn_feat_list]
        fused_fpn = torch.cat(upsampled_feature_maps, dim=1)
        

        """ Text Seg Head """
        text_feat = self.texthead(fused_fpn)

        return text_feat


class SalientTextModel(nn.Module):
    def __init__(self,cfg,input_size=896, device='cuda'):
        super(SalientTextModel, self).__init__()

        inner_channels = 1280
        self.input_size = input_size
        self.device = device

        self.backbone = build_backbone(cfg)
        
        self.upsample = nn.Upsample(size= None , mode='bilinear', align_corners=False)
        self.salhead = define_salD(inner_channels, netSalD='denseNet_15layer')
        self.low_salhead = define_salD(13, netSalD='denseNet_15layer')

        self.texthead = TextDetector(inner_channels)
        
        self.low_level = LowLevelFeat(img_size=self.input_size)
        self.sal_fused = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0)
        
    def step_function(self, x, y):
            return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))
        
    def forward(self, x):
        """ Base network """
        fpn_feat = self.backbone(x)
        
        """ Upsample and fuse FPN features"""
        fpn_feat_list= list(fpn_feat.values())
        max_height = max([fm.size(2) for fm in fpn_feat_list])
        max_width = max([fm.size(3) for fm in fpn_feat_list])
        self.upsample.size = [max_height, max_width] 
        upsampled_feature_maps = [self.upsample(fm) for fm in fpn_feat_list]
        fused_fpn = torch.cat(upsampled_feature_maps, dim=1)
        
        low_feat = self.low_level(x).to(self.device)

        """Sal Head """
        high_sal_feat = self.salhead(fused_fpn)
        low_sal_feat = self.low_salhead(low_feat)
        high_sal_feat = transforms.Resize(self.input_size)(high_sal_feat)
        low_sal_feat = transforms.Resize(self.input_size)(low_sal_feat)

        sal_feat_fused = self.sal_fused(torch.mul(low_sal_feat, high_sal_feat))

        """ Text Head """
        text_feat = self.texthead(fused_fpn)

        """ Refine features"""
                
        sal_feat = transforms.Resize(self.input_size)(sal_feat_fused)
        refine_text_feature = torch.mul(text_feat, sal_feat)

        return sal_feat,refine_text_feature
    
    
def build_text_model(backbone_cfg, input_size, device='cuda'):

    cfg = get_cfg()
    add_swint_config(cfg)
    cfg.merge_from_file(backbone_cfg)

    return  TextModel(cfg,input_size=input_size, device=device)


def build_salientText_model(backbone_cfg, input_size, device='cuda'):

    cfg = get_cfg()
    add_swint_config(cfg)
    cfg.merge_from_file(backbone_cfg)

    return  SalientTextModel(cfg,input_size=input_size, device=device)

