import os, glob, sys, re

import numpy as np
import chess
import pandas as pd

from importlib import reload

sys.path.append('../')

# create dict to map letters to numbers and vice versa (translate moves)
letter2num = {letter: num for num, letter in enumerate('abcdefgh', 1)}
num2letter = {num: letter for num, letter in enumerate('abcdefgh', 1)} 

def move2matrix(move, board):
    """
    Given a move and a board object, a matrix represent the position the piece was moved from to the position the piece was moved to.

    Parameters:
        move (chess.Move): chess.Move object
        board (chess.Board): chess.Board object
    
    Returns
        matrix (np.array): 8x8xn matrix representing the move
    """

    # get the move in uci format (e.g. 'e2e4')``
    board.push_san(move).uci()
    
    # extract individual move
    move = board.pop()

    # make the "from" move
    from_out = np.zeros(8,8)
    from_row = 8 - int(move[1]) # get move rank
    from_col = letter2num(move[0]) # get move file
    from_out[from_row, from_col] = 1

    # make the "to" move
    to_out = np.zeros(8,8)
    to_row = 8 - int(move[3])
    to_col = letter2num(move[2])
    to_out[to_row, to_col] = 1

    return np.stack([from_out, to_out])
     