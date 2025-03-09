import os, sys, glob
import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.append('../')

# custom imports
from utils.env import *
from utils.model import *

class ChessTrainer:
    pass
