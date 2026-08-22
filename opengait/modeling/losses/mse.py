import torch.nn.functional as F
import torch

from .base import BaseLoss
from pytorch_msssim import ssim

# class MSELoss(BaseLoss):
#     def __init__(self, loss_term_weight=1.0):
#         super(MSELoss, self).__init__(loss_term_weight)

#     def forward(self, student_feat, teacher_feat):
#         """
#             feat: [n, c, h, w]
#             teacher_feat: [n, c, h, w]
#         """
#         # distances= torch.mean(torch.norm(feat-teacher_feat,dim=2), dim=1)
#         # loss = torch.sum(distances)

#         loss = ((student_feat - teacher_feat) ** 2)
     #         self.info.update({'loss': loss.detach().clone()})
#         return loss, self.info

# class MSELoss(BaseLoss):
#     def __init__(self, loss_term_weight=1.0):
#         super(MSELoss, self).__init__(loss_term_weight)

#     def forward(self, reconsgtsil, gtsils):
#         """
#             feat: [n, c, h, w]
#             teacher_feat: [n, c, h, w]
#         """
#         # distances= torch.mean(torch.norm(feat-teacher_feat,dim=2), dim=1)
#         # loss = torch.sum(distances)

#         loss = ((reconsgtsil - gtsils) ** 2)
     #         self.info.update({'loss': loss.detach().clone()})
#         return loss, self.info

class MSELoss(BaseLoss):
    def __init__(self, loss_term_weight=1.0):
        super(MSELoss, self).__init__(loss_term_weight)
        self.loss_term_weight = loss_term_weight

    def ssim_loss(self, denoised, gt_body):
        ssim_val = ssim(denoised.float(), gt_body.float(), data_range=1.0, size_average=True)
        return 1 - ssim_val  # SSIM越大越好，因此Loss定义为1-SSIM

    def forward(self, denoised, x, iter=0, start_iter=0):
        """
            feat: [n, c, h, w]
            teacher_feat: [n, c, h, w]
        """
        # distances= torch.mean(torch.norm(feat-teacher_feat,dim=2), dim=1)
        # loss = torch.sum(distances)
             # loss = torch.mean((denoised - x) ** 2)
        diff = (denoised - x) ** 2
        mseloss = diff.mean(dim=list(range(1, len(diff.shape))))
        ssim_l = self.ssim_loss(denoised, x)

        if iter < start_iter:
            self.loss_term_weight = 0
        else:
            self.loss_term_weight = 1

        # loss = mseloss * self.loss_term_weight
        loss = (mseloss + 0.5 * ssim_l) * self.loss_term_weight

        self.info.update({'loss': loss.detach().clone()})
        return loss, self.info