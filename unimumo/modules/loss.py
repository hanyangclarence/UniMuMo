import torch
import torch.nn as nn


class MotionVqVaeLoss(nn.Module):
    def __init__(self, commitment_loss_weight: float = 1.0, motion_weight: float = 1.0):
        super().__init__()
        self.commitment_loss_weight = commitment_loss_weight
        self.motion_weight = motion_weight
        self.recon_loss = nn.MSELoss()

    def forward(self, motion_gt: torch.Tensor, motion_recon: torch.Tensor, commitment_loss: torch.Tensor, split: str = "train"):
        motion_rec_loss = self.recon_loss(motion_recon.contiguous(), motion_gt.contiguous())

        loss = self.motion_weight * motion_rec_loss + self.commitment_loss_weight * commitment_loss
        rec_loss = self.motion_weight * motion_rec_loss

        log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
               "{}/commitment_loss".format(split): commitment_loss.detach().mean(),
               "{}/rec_loss".format(split): rec_loss.detach().mean(),
               }
        return loss, log
