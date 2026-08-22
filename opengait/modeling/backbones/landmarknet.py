import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ..modules import SetBlockWrapper, BasicConv2d

import os
import time
import cv2
import json
import kornia

class FeatureExtractor(nn.Module):
    '''
    input: [n, c, s, h, w]
    outpot: [n, c, s, h/4, w/4]
    '''
    def __init__(self, model_cfg):
        super(FeatureExtractor, self).__init__()
        in_channels = model_cfg['channels']  # in_channels = [32, 64, 128]

        self.model = nn.Sequential(BasicConv2d(1, in_channels[0], 7, 1, 3),
                                    nn.LeakyReLU(inplace=True),
                                    BasicConv2d(in_channels[0], in_channels[0], 5, 1, 2),
                                    nn.LeakyReLU(inplace=True),
                                    BasicConv2d(in_channels[0], in_channels[1], 4, 2, 1),
                                    nn.LeakyReLU(inplace=True),
                                    BasicConv2d(in_channels[1], in_channels[1], 3, 1, 1),
                                    nn.LeakyReLU(inplace=True),
                                    BasicConv2d(in_channels[1], in_channels[2], 4, 2, 1),
                                    nn.LeakyReLU(inplace=True))
        self.model = SetBlockWrapper(self.model)

    def forward(self, x):
        return self.model(x)

class ParameterRegressor(nn.Module):
    def __init__(self, num_features, num_parts):
        super(ParameterRegressor, self).__init__()
        """
        convolutional encoder + linear layer at the end
        Args:
            num_features: list of ints containing number of features per layer
            num_parts: number of body parts for which we regress affine parameters
        Returns:
            torch.tensor (batch, num_parts, 2, 3), (2, 3) affine matrix for each body part
        """
        self.height, self.width = 64, 44
        self.num_features = num_features
        self.num_parts = num_parts
        # self.layers = self.define_network(num_features)
        self.model = nn.Sequential(BasicConv2d(1, num_features[0], 3, 1, 1),
                                    nn.BatchNorm2d(num_features[0]),
                                    nn.LeakyReLU(inplace=True),
                                    BasicConv2d(num_features[0], num_features[1], 3, 2, 1),
                                    nn.BatchNorm2d(num_features[1]),
                                    nn.LeakyReLU(inplace=True),
                                    BasicConv2d(num_features[1], num_features[2], 4, 2, 1),
                                    nn.BatchNorm2d(num_features[2]),
                                    nn.LeakyReLU(inplace=True),
                                    BasicConv2d(num_features[2], num_features[3], 3, 1, 1),
                                    nn.BatchNorm2d(num_features[3]),
                                    nn.LeakyReLU(inplace=True),
                                    )
        self.model = SetBlockWrapper(self.model)
        self.linear = nn.Sequential(
                                    nn.Linear(num_features[3]*int(self.height/4)*int(self.width/4),num_features[3]),
                                    nn.BatchNorm1d(num_features[3],num_features[3]),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Linear(num_features[3],self.num_parts*6)
        )
        def forward(self, x):
            n, c, s, h, w = x.size()
            feature = self.model(x)
            x = feature.view(n*s*c, -1)
            params = self.linear(x)

            return feature, params.view(n, self.num_parts, s,  2, 3)
    # def forward(self, input):
    #     return self.layers(input).view(-1, self.num_parts, 2, 3)

class SpatialSoftmax(torch.nn.Module):
    '''
    input: [n, c, s, h/4, w/4]
    feature_landmarks: [n, n_landmark, s, 2]
    '''
    def __init__(self, height, width, channel, lim=[-1., 1., -1., 1.], temperature=None):
        super(SpatialSoftmax, self).__init__()
        self.height = height
        self.width = width
        self.channel = channel

        if temperature:
            self.temperature = Parameter(torch.ones(1) * temperature)
        else:
            self.temperature = 1.

        pos_x, pos_y = np.meshgrid(
            np.linspace(lim[0], lim[1], self.width),
            np.linspace(lim[2], lim[3], self.height))

        pos_x = torch.from_numpy(pos_x.reshape(self.height * self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height * self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

    def forward(self, feature):
        n, c, s, h, w = feature.size()
        feature = feature.view(-1, self.height * self.width)

        softmax_attention = F.softmax(feature / self.temperature, dim=-1)
        expected_x = torch.sum(Variable(self.pos_x) * softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(Variable(self.pos_y) * softmax_attention, dim=1, keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y], 1)
        feature_landmarks = expected_xy.view(n, self.channel, s, 2)

        return feature_landmarks

class LandmarkPredictor_1(nn.Module):
    '''
    input: [n, 1, s, h, w]
    return: [n, c, s, h, w]
    '''
    def __init__(self, model_cfg):
        super(LandmarkPredictor_1, self).__init__()

        in_channels = model_cfg['channels']  # in_channels = [32, 64, 128]

        self.model = nn.Sequential(BasicConv2d(1, in_channels[0], 7, 1, 3),
                                    nn.LeakyReLU(inplace=True),
                                    BasicConv2d(in_channels[0], in_channels[0], 5, 1, 2),
                                    nn.LeakyReLU(inplace=True),
                                    BasicConv2d(in_channels[0], in_channels[1], 4, 2, 1),
                                    nn.LeakyReLU(inplace=True),
                                    BasicConv2d(in_channels[1], in_channels[1], 3, 1, 1),
                                    nn.LeakyReLU(inplace=True),
                                    BasicConv2d(in_channels[1], in_channels[2], 4, 2, 1),
                                    nn.LeakyReLU(inplace=True))

        self.model = SetBlockWrapper(self.model)

    def forward(self, x):
        return self.model(x)

class LandmarkPredictor_2(nn.Module):
    '''
    input: [n, c, s, h/4, w/4]
    return: [n, num_landmark, s, h/4, w/4]
    '''
    def __init__(self, model_cfg, num_landmark, lim=[-1., 1., -1., 1.]):
        super(LandmarkPredictor_2, self).__init__()

        in_channels = model_cfg['channels']  # in_channels = [32, 64, 128]
        img_height = model_cfg['height']
        img_width = model_cfg['width']

        self.model_landmark = nn.Sequential(BasicConv2d(in_channels[2], num_landmark, 1, 1, 0))
        self.model_landmark = SetBlockWrapper(self.model_landmark)

        self.integrater = SpatialSoftmax(
            height=img_height//4, width=img_width//4, channel=num_landmark, lim=lim)

    def integrate(self, heatmap):
        return self.integrater(heatmap)

    def forward(self, x):
        heatmap = self.model_landmark(x)
        return self.integrate(heatmap)

class Refiner(nn.Module):
    '''
    input: [n, num_landmark, s, h/4, w/4]
    return: [n, 1, s, h, w]
    '''
    def __init__(self, num_features, model_cfg):
        super(Refiner, self).__init__()
        in_channels = model_cfg['channels']
        self.num_features = num_features
        self.downmodel = nn.Sequential(BasicConv2d(19, num_features[0], 3, 1, 1),
                                    nn.BatchNorm2d(num_features[0]),
                                    nn.LeakyReLU(inplace=True),
                                    BasicConv2d(num_features[0], num_features[1], 3, 2, 1),
                                    nn.BatchNorm2d(num_features[1]),
                                    nn.LeakyReLU(inplace=True),
                                    BasicConv2d(num_features[1], num_features[2], 4, 2, 1),
                                    nn.BatchNorm2d(num_features[2]),
                                    nn.LeakyReLU(inplace=True),
                                    BasicConv2d(num_features[2], num_features[3], 3, 1, 1),
                                    nn.BatchNorm2d(num_features[3]),
                                    nn.LeakyReLU(inplace=True),
                                    )
        self.upmodel = nn.Sequential(# BasicConv2d(in_channels[3], in_channels[3], 4, 2, 1),
                                    nn.ConvTranspose2d(in_channels[2], in_channels[2], 4, 2, 1),
                                    nn.LeakyReLU(inplace=True),
                                    BasicConv2d(in_channels[2], in_channels[1], 3, 1, 1),
                                    nn.LeakyReLU(inplace=True),
                                    # BasicConv2d(in_channels[2], in_channels[2], 4, 2, 1),
                                    nn.ConvTranspose2d(in_channels[1], in_channels[1], 4, 2, 1),
                                    nn.LeakyReLU(inplace=True),
                                    BasicConv2d(in_channels[1], in_channels[0], 5, 1, 2),
                                    nn.LeakyReLU(inplace=True),
                                    BasicConv2d(in_channels[0], 1, 7, 1, 3))
        self.downmodel = SetBlockWrapper(self.downmodel)
        self.upmodel = SetBlockWrapper(self.upmodel)

    def forward(self, feat):
        feat = self.downmodel(feat)
        feat = self.upmodel(feat)
        return feat

class landmarknet(nn.Module):
    def __init__(self, landmark_cfg):
        super(landmarknet, self).__init__()

        model_cfg = landmark_cfg

        self.inv_std = model_cfg['inv_std']  # 10.0
        self.height = model_cfg['height']    # 64
        self.width = model_cfg['width']      # 44
        self.num_landmark = model_cfg['num_landmark']  # n_landmark
        self.freeze = model_cfg['freeze_half']  # fix or not
        self.template_path = model_cfg.get('template_path')
        if self.template_path is None:
            raise ValueError('landmark_cfg.template_path is required.')

        self.regressor = ParameterRegressor(num_features=model_cfg['regressor_nf'], num_parts=model_cfg['num_parts'])
        self.I = torch.eye(3)[0:2]
        # # visual feature extractor
        # self.extract_feature = FeatureExtractor(model_cfg)

        # # landmark predictor
        # self.extract_landmark_1 = LandmarkPredictor_1(model_cfg)
        # self.extract_landmark_2 = LandmarkPredictor_2(model_cfg, num_landmark=model_cfg['num_landmark'], lim=[-1., 1., -1., 1.])

        if model_cfg['freeze_half']:
            self.regressor.requires_grad_(False)
        #     self.extract_feature.requires_grad_(False)
        #     self.extract_landmark_1.requires_grad_(False)
        #     self.extract_landmark_2.requires_grad_(False)

        # # map the feature back to the image
        # self.refine = Refiner(model_cfg['regressor_nf'],model_cfg)

    def landmark2heatmap(self, landmark, inv_std=10.):
        # landmark: N x n_landmark x 2
        # heatpmap: N x n_landmark x (H / 4) x (W / 4)
        # return: N x n_landmark x (H / 4) x (W / 4)
        height = self.height // 4
        width = self.width // 4
        mu_x, mu_y = landmark[:, :, :, :1].unsqueeze(-1), landmark[:, :, :, 1:].unsqueeze(-1)

        y = (torch.linspace(-1.0, 1.0, height).view(1, 1, 1, height, 1).to(mu_y.device)).detach() # H
        x = (torch.linspace(-1.0, 1.0, width).view(1, 1, 1, 1, width).to(mu_x.device)).detach() # W

        g_y = (y - mu_y)**2
        g_x = (x - mu_x)**2
        dist = (g_y + g_x) * inv_std**2

        hmap = torch.exp(-dist)

        return hmap

    def transport(self, src_feat, des_feat, src_hmap, des_hmap, des_feat_hmap=None):
        # src_feat: N x C × S x (H / 4) x (W / 4)
        # des_feat: N x C × S x (H / 4) x (W / 4)
        # src_hmap: N x n_landmark × S x (H / 4) x (W / 4)
        # des_hmap: N x n_landmark × S x (H / 4) x (W / 4)
        # des_feat_hmap = des_hmap * des_feat: N x C x (H / 4) * (W / 4)
        # mixed_feat: N x C × S x (H / 4) x (W / 4)
        src_hmap = torch.sum(src_hmap, 1, keepdim=True)
        des_hmap = torch.sum(des_hmap, 1, keepdim=True)
        src_digged = src_feat * (1. - src_hmap) * (1. - des_hmap)

        if des_feat_hmap is None:
            mixed_feat = src_digged + des_hmap * des_feat
        else:
            mixed_feat = src_digged + des_feat_hmap

        return mixed_feat

    def mixfeature(self, src, des_hmap):
        # src: N x C × S x H x W
        # des_hmap: N x n_landmark × S x H x W
        # mixed_feat: N x C × S x (H / 4) x (W / 4)
        mixed_feat = torch.cat((src,des_hmap),dim=1)

        return mixed_feat

    def draw_shape(self,pos, sigma_x, sigma_y, angle, size):
        """
        draw (batched) gaussian with sigma_x, sigma_y on 2d grid

        Args:
            pos: torch.tensor (float) with shape (2) specifying center of gaussian blob (x: row, y:column)
            sigma_x: torch.tensor (float scalar), scaling parameter along x-axis
            sigma_y: similar along y-axis
            angle: torch.tensor (float scalar) rotation angle in radians
            size: int specifying size of image
            device: torch.device, either cpu or gpu

        Returns:
            torch.tensor (1, 1, size, size) with gaussian blob
        """
        device = pos.device
        assert sigma_x.device == sigma_y.device == angle.device == device, "inputs should be on the same device!"

        # create 2d meshgrid
        x, y = torch.meshgrid(torch.arange(0, size[0]), torch.arange(0, size[1]), indexing='ij')
        x, y = x.unsqueeze(0).unsqueeze(0).to(device), y.unsqueeze(0).unsqueeze(0).to(device)

        # see https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function
        a = torch.cos(angle) ** 2 / (2 * sigma_x ** 2) + torch.sin(angle) ** 2 / (2 * sigma_y ** 2)
        b = -torch.sin(2 * angle) / (4 * sigma_x ** 2) + torch.sin(2 * angle) / (4 * sigma_y ** 2)
        c = torch.sin(angle) ** 2 / (2 * sigma_x ** 2) + torch.cos(angle) ** 2 / (2 * sigma_y ** 2)

        # append dimsensions for broadcasting
        pos = pos.view(1, 1, 2, 1, 1)
        a, b, c = a.view(1, 1), b.view(1, 1), c.view(1, 1)

        # pixel-wise distance from center
        xdist = (x - pos[:, :, 0])
        ydist = (y - pos[:, :, 1])

        # gaussian function
        g = torch.exp((-a * xdist ** 2 - 2 * b * xdist * ydist - c * ydist ** 2))

        return g

    def draw_template(self,path, size, batch_size, device):
        """
        draw template consisting of limbs defined by gaussian heatmap
        Args:
            template: json file defining all parts
            size: int, image size (assumed quadratic), this should match the center coordinates defined in the json!
            device: torch.device, either cpu or gpu
        """
        n, c, s, h, w = batch_size
        with open(path, 'r') as file:
            template = json.load(file)
        heatmaps = []
        if size == [16,11]:
            for v in template.values():
                center = torch.tensor([v['center'][0]//4,v['center'][1]//4]).to(device)
                sx = torch.tensor(v['sx']/4).to(device)
                sy = torch.tensor(v['sy']/4).to(device)
                angle = torch.tensor(v['angle']).to(device)
                heatmaps.append(self.draw_shape(center, sx, sy, angle, size))
        else:
            for v in template.values():
                center = torch.tensor([v['center'][0],v['center'][1]]).to(device)
                sx = torch.tensor(v['sx']).to(device)
                sy = torch.tensor(v['sy']).to(device)
                angle = torch.tensor(v['angle']).to(device)
                heatmaps.append(self.draw_shape(center, sx, sy, angle, size))
        # img = torch.cat(heatmaps, dim=1).sum(1)[0].detach().cpu().numpy()*255
        # cv2.imwrite('heatmap.png',img)
        heatmaps = torch.cat(heatmaps, dim=1).unsqueeze(2).repeat(n, 1, s, 1, 1)

        return heatmaps

    def transform_template(self,input, params):
        h,w = input.shape[-2:]
        # scale up translation
        params[..., -1] = params[..., -1] * torch.tensor([h,w]).to(input.device)
        return kornia.geometry.warp_affine(input, params, dsize=(h, w))

    def forward(self, sils):
        n, c, s, h, w = sils.size()
        des = sils  # [n, c=1, s, h, w]
        src = torch.roll(sils, shifts=8, dims=2)  # [n, c=1, s, h, w]
        batched_template = self.draw_template(self.template_path, size=[64, 44], batch_size=sils.shape, device=sils.device)
        # set_trace()
        # img = batched_templatevis.sum(1)[0].detach().cpu().numpy()*255
        # cv2.imwrite('heatmap.png',img[0])
        if self.freeze:
            with torch.no_grad():
                des_feat, des_params = self.regressor(des)
                des_params = self.I.to(sils.device) + des_params
                batched_template = batched_template.view(-1, h, w).unsqueeze(1)
                batched_des_params = des_params.view(-1, 2, 3)
        else:
            src_feat, src_params = self.regressor(src)
            src_feat = src_feat.view(n, -1, s, int(h/4), int(w/4))
            src_params = self.I.to(sils.device) + src_params
            des_feat, des_params = self.regressor(des)
            des_feat = des_feat.view(n, -1, s, int(h/4), int(w/4))
            des_params = self.I.to(sils.device) + des_params

            batched_template = batched_template.view(-1, h, w).unsqueeze(1)
            # batched_templatevis = batched_templatevis.view(-1, h, w).unsqueeze(1)
            batched_src_params = src_params.view(-1, 2, 3)
            batched_des_params = des_params.view(-1, 2, 3)
                     # transformed_template = self.transform_template(batched_template, batched_params).view(n, 18, s, h, w)
                     # src_feat = self.extract_feature(src)  # [n, c, s, h/4, w/4]
            # des_feat = self.extract_feature(des)  # [n, c, s, h/4, w/4]
            # src_landmark = self.extract_landmark_1(src)
            # des_landmark = self.extract_landmark_1(des)
            # src_landmark = self.extract_landmark_2(src_landmark)  # [n, n_landmark, s, 2]
            # des_landmark = self.extract_landmark_2(des_landmark)  # [n, n_landmark, s, 2]

        # src_hmap = self.transform_template(batched_template, batched_src_params).view(n, 18, s, h, w)  # [n, n_landmark, s, h, w]
        des_hmap = self.transform_template(batched_template, batched_des_params).view(n, 18, s, h, w)  # [n, n_landmark, s, h, w]
        # src_hmap = self.transform_template(batched_template, batched_src_params).view(n, 18, s, h, w)  # [n, n_landmark, s, h, w]
        # des_hmapvis = self.transform_template(batched_templatevis, batched_des_params).view(n, 18, s, h, w)  # [n, n_landmark, s, h, w]
        # set_trace()
        # img = src_hmap.sum(1)[0].detach().cpu().numpy()*255
        # cv2.imwrite('heatmap.png',img[0])

        # src_hmap = self.landmark2heatmap(src_landmark, self.inv_std)  # [n, n_landmark, s, h/4, w/4]
        # des_hmap = self.landmark2heatmap(des_landmark, self.inv_std)  # [n, n_landmark, s, h/4, w/4]
        # mixed_feat = self.transport(src_feat, des_feat, src_hmap, des_hmap)  # [n, c, s, h/4, w/4]
        # mixed_feat = self.mixfeature(src, des_hmap)  # [n, c, s, h/4, w/4]
             # des_pred = self.refine(mixed_feat)  # [n, c=1, s, h, w]
             # return des, des_pred, des_hmap, src_hmap, des_feat, src_feat, des_landmark
        # return des, des_pred, des_hmap
        return  des_hmap
        # return des, des_pred, des_hmap, src_hmap, des_feat, src_feat
        # 剪影，重建剪影，目标剪影heatmap，原剪影heatmap，目标剪影的特征，原始剪影特征，
