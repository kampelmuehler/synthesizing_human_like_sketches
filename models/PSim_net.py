import torch.nn as nn
from models.PSim_alexnet import PSim_Alexnet
import torch
from utils import utils


class PSimNet(nn.Module):
    """Pre-trained network with all channels equally weighted by default (cosine similarity)"""

    def __init__(self, device=torch.device("cuda:0")):
        super(PSimNet, self).__init__()
        checkpoint_path = 'pretrained_models/PSim_alexnet.pt'
        self.net = PSim_Alexnet(train=False)
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        self.net.load_weights(checkpoint['state_dict'])
        self.net.to(device)
        # freeze network
        for param in self.net.parameters():
            param.requires_grad = False
        self.net.eval()

    def forward(self, generated, target):
        outs0 = self.net.forward(generated)
        outs1 = self.net.forward(target)

        for (kk, out0) in enumerate(outs0):
            cur_score = torch.mean((1. - utils.cos_sim(outs0[kk], outs1[kk])))  # mean is over batch
            if kk == 0:
                val = 1. * cur_score
            else:
                val = val + cur_score

        return val
