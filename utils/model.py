import chess
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ChessNet(nn.Module):
    """
    simple chess network - conv + policy head + value head
    """

    def __init__(self):
        super(ChessNet, self).__init__()

        # convolutional block
        self.conv_block = nn.Sequential(
            nn.Conv2d(14, 256, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # policy head - what move to make?
        self.policy_head = nn.Sequential(
            # convert channels to 73 (num squares + num promotions)
            nn.Conv2d(256, 73, kernel_size=1),
            nn.Flatten(),
            nn.Linear(73*8*8, 4672) # 8 x 8 x 73 = 4672
        )

        # value head - who is winning?
        self.value_head = nn.Sequential(
            nn.Conv2d(256, 1, 1),
            nn.Flatten(),
            nn.Linear(8*8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # fwd, get move and evaluation
        x = self.conv_block(x)
        p = self.policy_head(x)
        v = self.value_head(x)
        return p, v