import os, sys, glob
import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class ChessEnv:
    """
    A simple environment for returning states, steps, and actions
    """

    def __init__(self):
        """Create the board"""
        self.board = chess.Board()

    def reset(self):
        """Reset the board"""
        self.board.reset()
        return self.get_state()
    
    def get_state(self):
        """Returns the state of the board as an 8x8 matrix"""
        board_state = []
        
        # loop through squares
        for square in chess.SQUARES:

            # get the piece on the square
            piece = self.board.piece_at(square)

            if piece:
                # use integers to represent pieces - positive for white, negative for black
                board_state.append(piece.piece_type if piece.color == chess.WHITE else -piece.piece_type)
            else:
                # use 0 to represent empty squares
                board_state.append(0)
        
        # board is a list of 64 integers representing the board state
        return board_state
    
    def step(self, action):
        """Executes an action (mode) and returns the next state, reward, and game status (done or not)"""

        # make move
        move = self.board.legal_moves[action]
        self.board.push(move)
        
        # init reward and done
        reward = 0
        done = False

        # check game status
        # loss
        if self.board.is_checkmate():
            reward = -1
            done = True
        # draw
        elif self.board.is_stalemate() or self.board.is_insufficient_material():
            reward = 0
            done = True
        # check
        elif self.board.is_check():
            reward = 0.5

        return self.get_state(), reward, done
    
    def get_legal_moves(self):
        """Return a list of legal moves"""
        return list(self.board.legal_moves)

class ChessDataset(Dataset):

    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass