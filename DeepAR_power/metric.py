import torch
import torch.nn as nn

class NGLLLoss(nn.Module):
    def __init__(self,):
        super(NGLLLoss, self).__init__()

    def forward(self, mu, sigma, target):

        likelihood = (1/2)*((target-mu)/sigma)**2 + (1/2)*torch.log(torch.tensor(2*torch.pi)) + torch.log(sigma)

        # another way
        # dist = torch.distributions.normal.Normal(mu_seq, sigma_seq)
        # -torch.sum(dist.log_prob(target))

        return torch.sum(likelihood)
    

def smape(true, pred):

    v = 2 * abs(pred - true) / (abs(pred) + abs(true))
    output = torch.mean(v) * 100

    return output
    