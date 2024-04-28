import os, glob, sys, re

import numpy as np
import chess
import pandas as pd

from importlib import reload

sys.path.append('../')

def board2matrix(board):
    """
    Given a chess.Board object, return a 8x8 matrix representing the board.

    Parameters:
        board (chess.Board): chess.Board object
    Returns
        board_matrix (np.array): 8x8 matrix representing the board
    """

    pieces = ['p', 'r', 'n', 'b', 'q', 'k']
    layers = []
    for piece in pieces:
        layers.append(create_matrix_layer(board, piece))
    board_matrix = np.stack(layers, axis=0)

    return board_matrix

def create_matrix_layer(board, type):

    """
    Given a chess.Board object and a piece type, return a 8x8 matrix representing the piece positions.

    Parameters:
        board (chess.Board): chess.Board object
        type (str): piece type
    
    Returns
        layer (np.array): 8x8 matrix representing the piece positions
    """

    # convert board to string (each piece is a letter, each space is a dot)
    s = str(board)

    # remove all chars that are not equal to the piece type, and make white pieces uppercase
    s = re.sub(f'[{type}{type.upper()} \n]', '.', s)

    # replace black pieces with -1
    s = re.sub(f'{type}', '-1', s)

    # replace white pieces with 1
    s = re.sub(f'{type.upper()}', '1', s)

    # replace empty spaces with 0
    s = re.sub('\.', '0', s)

    # make actual matrix
    board_matrix = []

    # loop through each row
    for row in s.split('\n'):

        # look at each val
        row = row.split(' ')

        # convert str to int
        row = [int(x) for x in row]

        # append to matrix
        board_matrix.append(row)
    
    return np.array(board_matrix)