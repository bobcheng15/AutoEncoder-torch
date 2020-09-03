import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.dconv1 = nn.ConvTranspose2d(64, 64, 3, 1, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.dconv2 = nn.ConvTranspose2d(64, 32, 4, 2, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.dconv3 = nn.ConvTranspose2d(32, 1, 8, 4, padding=0)
        self.bn3   = nn.BatchNorm2d(1)
        self.relu  = nn.ReLU()
    
    def forward(self, x):
        x = self.dconv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dconv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dconv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = torch.clamp(x, min=0.0, max=1.0)
        return x 


