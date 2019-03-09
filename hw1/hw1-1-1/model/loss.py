import torch.nn.functional as F
import torch.nn as nn

def nll_loss(output, target):
    return F.nll_loss(output, target)

def L2_loss(output, target):
	criterion = nn.MSELoss() 
	return criterion(output, target)
