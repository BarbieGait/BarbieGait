import torch.nn as nn
import torch.nn.functional as F
import torch

from .base import BaseLoss

# ==============================蒸馏损失===============================
class DistillKL(BaseLoss):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T=2.0):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_t, y_s):
        # student网络输出软化后结果
        # log_softmax与softmax没有本质的区别，只不过log_softmax会得到一个正值的loss结果。
        p_s = F.log_softmax(y_s/self.T, dim=1)

        # # teacher网络输出软化后结果
        p_t = F.softmax(y_t/self.T, dim=1)

        # 蒸馏损失采用的是KL散度损失函数
        # loss = F.kl_div(p_s, p_t, reduction='mean') * (self.T**2) / y_s.shape[0]
        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T**2)

        self.info.update({'loss': loss.detach().clone()})
        return loss, self.info
