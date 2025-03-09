import chess
import numpy as np

class ChessEnv:
    """
    A simple environment for returning states, steps, and actions
    """

    def __init__(self):
        self.board = chess.Board()
    
    def reset(self):
        self.board = chess.Board()
        return self.board