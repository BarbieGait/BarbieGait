import torch.nn.functional as F
import torch

from .base import BaseLoss

class NoiseMaskLoss(BaseLoss):
    def __init__(self, loss_term_weight=1.0):
        super(NoiseMaskLoss, self).__init__(loss_term_weight)

    def forward(self, noise_activate):
        """
            feat: [n, c, h, w]
            teacher_feat: [n, c, h, w]
        """
        # distances= torch.mean(torch.norm(feat-teacher_feat,dim=2), dim=1)
        # loss = torch.sum(distances)
        n, c, s, h, w = noise_activate.shape
        noise_activate = noise_activate.reshape(n,c,s,-1)
        loss = torch.sum(noise_activate,dim=-1).mean(-1)
        self.info.update({'loss': loss.detach().clone()})
        return loss, self.info
