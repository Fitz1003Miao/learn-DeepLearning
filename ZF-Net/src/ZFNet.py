import torch
import torch.nn as nn

class ZFNet(nn.Module):
    def __init__(self, num_classes = 2):
        