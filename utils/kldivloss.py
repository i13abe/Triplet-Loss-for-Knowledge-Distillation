import torch
import torch.nn as nn
import torch.nn.functional as F



class KLDivLoss(nn.KLDivLoss):
    def forward(self, inputs, targets):
        return F.kl_div(
            F.log_softmax(inputs, dim=1),
            F.softmax(targets, dim=1),
            reduction=self.reduction,
            log_target=self.log_target,
        )
