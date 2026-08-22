import torch
import torch.nn as nn

from ..base_model import BaseModel
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateBNNecks, conv1x1, conv3x3,\
BasicBlockP3D_GON, BasicBlock2D_GON, GONFC, BasicBlock3D_GON

from einops import rearrange

blocks_map = {
    'gon_2d': BasicBlock2D_GON,
    'gon_p3d': BasicBlockP3D_GON,
    'gon_3d': BasicBlock3D_GON,
}

class GaitCLIF(BaseModel):

    def build_network(self, model_cfg):

        mode = model_cfg['Backbone']['mode']
        assert mode in blocks_map.keys()
        block = blocks_map[mode]

        in_channels = model_cfg['Backbone']['in_channels']
        layers      = model_cfg['Backbone']['layers']
        channels    = model_cfg['Backbone']['channels']
        self.inference_use_emb2 = model_cfg['use_emb2'] if 'use_emb2' in model_cfg else False

        if mode == '3d' or mode == 'gon_3d':
            strides = [
                [1, 1],
                [1, 2, 2],
                [1, 2, 2],
                [1, 1, 1]
            ]
        else:
            strides = [
                [1, 1],
                [2, 2],
                [2, 2],
                [1, 1]
            ]

        self.inplanes = channels[0]


        self.layer0 = SetBlockWrapper(nn.Sequential(
            conv3x3(in_channels, self.inplanes, 1),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True)
        ))
        part_num = model_cfg['partnum']
        self.layer1 = SetBlockWrapper(self.make_layer(BasicBlock2D_GON, channels[0], strides[0], blocks_num=layers[0], lnsize=[64,64,44], part_num=part_num[0], mode=mode))

        self.layer2 = self.make_layer(block, channels[1], strides[1], blocks_num=layers[1], lnsize=[128,32,22], part_num=part_num[1], mode=mode)
        self.layer3 = self.make_layer(block, channels[2], strides[2], blocks_num=layers[2], lnsize=[256,16,11], part_num=part_num[2], mode=mode)
        self.layer4 = self.make_layer(block, channels[3], strides[3], blocks_num=layers[3], lnsize=[512,16,11], part_num=part_num[3], mode=mode)
        if mode == 'gon_2d':
            self.layer2 = SetBlockWrapper(self.layer2)
            self.layer3 = SetBlockWrapper(self.layer3)
            self.layer4 = SetBlockWrapper(self.layer4)

        self.gon_fc = GONFC(16, channels[3], channels[2], norm=True, plnnum=part_num[-1])
        self.BNNecks = SeparateBNNecks(16, channels[2], class_num=model_cfg['SeparateBNNecks']['class_num'])

        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=[16])

    def make_layer(self, block, planes, stride, blocks_num, part_num=16, lnsize=None, mode='2d'):

        if max(stride) > 1 or self.inplanes != planes * block.expansion:
            if mode == 'gon_3d':
                downsample = nn.Sequential(nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=[1, 1, 1], stride=stride, padding=[0, 0, 0], bias=False), nn.BatchNorm3d(planes * block.expansion))
            elif mode == 'gon_2d':
                downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride=stride), nn.BatchNorm2d(planes * block.expansion))
            elif mode == 'gon_p3d':
                downsample = nn.Sequential(nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=[1, 1, 1], stride=[1, *stride], padding=[0, 0, 0], bias=False), nn.BatchNorm3d(planes * block.expansion))
            else:
                raise TypeError('xxx')
        else:
            downsample = lambda x: x

        layers = [block(self.inplanes, planes, stride=stride, lnsize=lnsize, part_num=part_num, downsample=downsample)]
        self.inplanes = planes * block.expansion
        s = [1, 1] if mode in ['gon_2d', 'gon_p3d'] else [1, 1, 1]
        for i in range(1, blocks_num):
            layers.append(
                    block(self.inplanes, planes, stride=s, lnsize=lnsize, part_num=part_num,)
            )
        return nn.Sequential(*layers)

    def forward(self, inputs):
        ipts, labs, typs, vies, seqL = inputs

        if len(ipts[0].size()) == 4:
            sils = ipts[0].unsqueeze(1)
        else:
            sils = ipts[0]
            sils = sils.transpose(1, 2).contiguous()
        assert sils.size(-1) in [44, 88]

        del ipts
        out0 = self.layer0(sils)

        out1 = self.layer1(out0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3) # [n, c, s, h, w]
             # Temporal Pooling, TP
        outs = self.TP(out4, seqL, options={"dim": 2})[0]  # [n, c, h, w]

        # Horizontal Pooling Matching, HPM
        feat = self.HPP(outs)  # [n, c, p]

        embed_1 = self.gon_fc(feat)  # [n, c, p]
        embed_2, logits = self.BNNecks(embed_1)  # [n, c, p]

        if self.inference_use_emb2:
                embed = embed_2
        else:
                embed = embed_1

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs},
            },
            'visual_summary': {
                'image/sils': rearrange(sils, 'n c s h w -> (n s) c h w'),
            },
            'inference_feat': {
                'embeddings': embed
            }
        }

        return retval
