import chess
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# res block
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )
    
    def forward(self, x):
        return F.relu(x + self.conv(x))

# chess network
class ChessNet(nn.Module):
    """
    simple chess network - conv + policy head + value head
    """

    def __init__(self):
        super(ChessNet, self).__init__()

        # convolutional block
        self.conv_block = nn.Sequential(
            nn.Conv2d(15, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
        )
        
        # policy head - what move to make?
        self.policy_head = nn.Sequential(
            # convert channels to 73 (num squares + num promotions)
            nn.Conv2d(256, 73, kernel_size=1),
            nn.Flatten(start_dim=1) # keep spatial dimension
        )

        # value head - who is winning?
        self.value_head = nn.Sequential(
            nn.Conv2d(256, 1, 1),
            nn.Flatten(),
            nn.Linear(8*8, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        # fwd, get move and evaluation
        x = self.conv_block(x)
        p = self.policy_head(x)
        v = self.value_head(x)
        return p, v