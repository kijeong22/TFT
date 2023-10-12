import torch
import torch.nn as nn

class QuantileLoss(nn.Module):
    def __init__(self, quantiles:list):
        super(QuantileLoss, self).__init__()

        self.quantiles = quantiles    

    def forward(self, true, pred):

        loss_set = []

        for i, q in enumerate(self.quantiles):
        
            error = true - pred[:,:,i]
            ql = torch.max(q*error, (q-1)*error)
            loss_set.append(ql)

        loss = torch.stack(loss_set, dim=-1) 
        loss = torch.mean(loss)

        return loss 
        