import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

from ..base_model import BaseModel
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks, conv1x1, conv3x3, BasicBlock2D, BasicBlockP3D, BasicBlock3D

from einops import rearrange
import cv2
from utils import get_valid_args, is_list, is_dict, np2var, ts2np, list2var, get_attr_from
from .. import models
from utils import config_loader, get_ddp_module, init_seeds, params_count, get_msg_mgr

blocks_map = {
    '2d': BasicBlock2D,
    'p3d': BasicBlockP3D,
    '3d': BasicBlock3D
}

class Encoder(nn.Module):
    def __init__(self,in_channels, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, latent_dim, kernel_size=4, stride=2, padding=1)
        # self.conv3 = nn.Conv2d(128, latent_dim, kernel_size=4, stride=2, padding=1)
    def forward(self, x):
        batch_size, c, s, h, w = x.shape
        x = x.view(batch_size * s, c, h, w)  # Merge batch and sequence dimensions
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim, out_channels):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(latent_dim, 64, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1)
        # self.deconv3 = nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1)
    def forward(self, x):
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        # x = F.relu(self.deconv3(x))
        # x = torch.sigmoid(self.deconv4(x))  # 用sigmoid来确保输出在0到1之间
        return x

class Autoencoder(nn.Module):
    def __init__(self, in_channels, latent_dim, out_channels):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(latent_dim, out_channels)
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

class recons_gtsil(BaseModel):
    def build_network(self, model_cfg):
        mode = model_cfg['Backbone']['mode']
        assert mode in blocks_map.keys()
        block = blocks_map[mode]

        in_channels = model_cfg['Backbone']['in_channels']
        layers      = model_cfg['Backbone']['layers']
        channels    = model_cfg['Backbone']['channels']
        self.inference_use_emb2 = model_cfg['use_emb2'] if 'use_emb2' in model_cfg else False

        if mode == '3d':
            strides = [
                [1, 1],
                [1, 2, 2],
                [1, 1, 1]
            ]
        else:
            strides = [
                [1, 1],
                [2, 2],
                [1, 1]
            ]

        self.inplanes = channels[0]
        self.layer0 = SetBlockWrapper(nn.Sequential(
            conv3x3(in_channels, self.inplanes, 1),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True)
        ))
        self.layer1 = SetBlockWrapper(self.make_layer(BasicBlock2D, channels[0], strides[0], blocks_num=layers[0], mode=mode))

        self.layer2 = self.make_layer(block, channels[1], strides[1], blocks_num=layers[1], mode=mode)
        self.layer3 = self.make_layer(block, channels[2], strides[2], blocks_num=layers[2], mode=mode)
        self.layer4 = self.make_layer(block, channels[3], strides[3], blocks_num=layers[3], mode=mode)

        if mode == '2d':
            self.layer2 = SetBlockWrapper(self.layer2)
            self.layer3 = SetBlockWrapper(self.layer3)
            self.layer4 = SetBlockWrapper(self.layer4)

        self.FCs = SeparateFCs(16, channels[3], channels[2])
        self.BNNecks = SeparateBNNecks(16, channels[2], class_num=model_cfg['SeparateBNNecks']['class_num'])

        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=[16])

        self.autoencoder = Autoencoder(1, 256, 1)

    def make_layer(self, block, planes, stride, blocks_num, mode='2d'):

        if max(stride) > 1 or self.inplanes != planes * block.expansion:
            if mode == '3d':
                downsample = nn.Sequential(nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=[1, 1, 1], stride=stride, padding=[0, 0, 0], bias=False), nn.BatchNorm3d(planes * block.expansion))
            elif mode == '2d':
                downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride=stride), nn.BatchNorm2d(planes * block.expansion))
            elif mode == 'p3d':
                downsample = nn.Sequential(nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=[1, 1, 1], stride=[1, *stride], padding=[0, 0, 0], bias=False), nn.BatchNorm3d(planes * block.expansion))
            else:
                raise TypeError('xxx')
        else:
            downsample = lambda x: x

        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * block.expansion
        s = [1, 1] if mode in ['2d', 'p3d'] else [1, 1, 1]
        for i in range(1, blocks_num):
            layers.append(
                    block(self.inplanes, planes, stride=s)
            )
        return nn.Sequential(*layers)

    def inputs_pretreament(self, inputs):
        ### Ensure the same data augmentation for heatmap and silhouette
        pose_sils = inputs[0]
        new_data_list = []
        for pose, sil in zip(pose_sils[0], pose_sils[1]):
            sil = sil[:, np.newaxis, ...] # [T, 1, H, W]
            pose = pose[:, np.newaxis, ...] # [T, 1, H, W]
            pose_h, pose_w = pose.shape[-2], pose.shape[-1]
            sil_h, sil_w = sil.shape[-2], sil.shape[-1]
            if sil_h != sil_w and pose_h == pose_w:
                cutting = (sil_h - sil_w) // 2
                pose = pose[..., cutting:-cutting]
            cat_data = np.concatenate([pose, sil], axis=1) # [T, 3, H, W]
            new_data_list.append(cat_data)
        new_inputs = [[new_data_list], inputs[1], inputs[2], inputs[3], inputs[4]]
        return super().inputs_pretreament(new_inputs)

    def forward(self, inputs):
        ipts, labs, typs, vies, seqL = inputs
        # teacher_outs = self.teacher(inputs)

        data = ipts[0]
        data = data.transpose(1, 2).contiguous()
        assert data.size(-1) in [44, 48, 88, 96]

        gtsils = data[:, :1, ...].contiguous()
        predsils = data[:, -1, ...].unsqueeze(1).contiguous()

        batch_size, c, s, h, w = predsils.shape
        reconsgtsil = self.autoencoder(predsils)
        reconsgtsil = reconsgtsil.view(batch_size, c, s, h, w)
        del ipts

        retval = {
            'training_feat': {
                # 'triplet': {'embeddings': embed_1, 'labels': labs},
                # 'distill': {'y_t': teacher_outs, 'y_s': logits},
                # 'softmax': {'logits': teacher_outs, 'labels': labs},
                # 'softmax': {'logits': logits, 'labels': labs},
                'mse': {'reconssil': reconsgtsil, 'gtsils': gtsils},
            },
            'visual_summary': {
                'image/gtsils': rearrange(gtsils * 255., 'n c s h w -> (n s) c h w'),
                'image/predsils': rearrange(predsils * 255., 'n c s h w -> (n s) c h w'),
                'image/reconssils': rearrange(reconsgtsil * 255., 'n c s h w -> (n s) c h w'),
            },
            # 'inference_feat': {
            #     'embeddings': embed
            # },
            # 'visual_feat': {
            #     'outs': out4,
            #     'sils': predsils
            # }
        }

        return retval
