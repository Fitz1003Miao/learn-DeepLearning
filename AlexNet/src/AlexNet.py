import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, num_classes = 2):
        super().__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(3, 96, 11, 4, padding = 2, groups = 1),
                                    nn.ReLU(inplace = True),
                                    nn.LocalResponseNorm(5, 1e-04, 0.75, 2),
                                    nn.MaxPool2d(3, 2))

        self.layer2 = nn.Sequential(nn.Conv2d(96, 256, 5, 1, padding = 2, groups = 2),
                                    nn.ReLU(inplace = True),
                                    nn.LocalResponseNorm(5, 1e-04, 0.75, 2),
                                    nn.MaxPool2d(3, 2))

        self.layer3 = nn.Sequential(nn.Conv2d(256, 384, 3, 1, padding = 1, groups = 1),
                                    nn.ReLU(inplace = True))

        self.layer4 = nn.Sequential(nn.Conv2d(384, 384, 3, 1, padding = 1, groups = 2),
                                    nn.ReLU(inplace = True))

        self.layer5 = nn.Sequential(nn.Conv2d(384, 256, 3, 1, padding = 1, groups = 2),
                                    nn.ReLU(inplace = True),
                                    nn.MaxPool2d(3, 2))

        self.layer6 = nn.Sequential(nn.Linear(256 * 6 * 6, 4096),
                                    nn.ReLU(inplace = True),
                                    nn.Dropout())

        self.layer7 = nn.Sequential(nn.Linear(4096, 4096),
                                    nn.ReLU(inplace = True),
                                    nn.Dropout())

        self.layer8 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x)))))
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.layer8(self.layer7(self.layer6(x)))
        return x

