import chess
import numpy as np

class ChessEnv:
    """
    environment for with board state representation + move masking
    """

    def __init__(self):
        self.board = chess.Board()
        self.move_converter = MoveConverter()
    
    def reset(self):
        self.board = chess.Board()
        return self.get_state(), self.get_mask()
    
    def get_state(self):
        """
        gets a 14 x 8 x 8 board state representation (14 channels for pieces + "special" moves, 8x8 board)
        """

        # init state representation
        state = np.zeros((14, 8, 8), dtype=np.float32)
        
        # loop through squares to fill in state representation
        for square in chess.SQUARES:
            
            # get piece on square
            piece = self.board.piece_at(square)

            # if piece exists, place it in appropriate channel
            if piece:
                channel = self.piece_to_channel(piece)
                row, col = self.square_to_coords(square)
                state[channel, row, col] = 1
            
        # add repetition + castling info
        state[12] = self.board.is_repetition(2)
        state[13] = self.board.has_castling_rights(chess.WHITE)
        state[14] = self.board.has_castling_rights(chess.BLACK)
        
        return state

    def get_mask(self):
        """
        gets a 8 x 8 mask of legal moves for the current player
        """

        # init mask
        mask = np.zeros(4672, dtype=np.float32) # why 4672?

        # loop through legal moves to fill in mask
        for move in self.board.legal_moves:
            idx = self.move_converter.move_to_idx(move)
            if idx is not None:
                mask[idx] = 1
        return mask
    
    def step(self, action):
        """
        Takes a step in the environment (makes a move)
        """

        # get move from action
        move = self.move_converter.idx_to_move(action)

        # if move is illegal, return state, mask, -1 reward
        if move not in self.board.legal_moves:
            return self.get_state(), self.get_mask(), -1, True
        
        # make move, init reward, check if game is over
        self.board.push(move)
        reward = 0
        done = self.board.is_game_over()

        # if game is over, get result and reward
        if done:
            result = self.board.result()
            reward = self.calculate_reward(result)
        
        # return state, mask, reward, done
        return self.get_state(), self.get_mask(), reward, done
    
    def calculate_reward(self, result):
        """
        given a game result, calculate the reward
        """

        # if white wins return 1 if white, -1 if black
        if result == "1-0":
            return 1 if self.board.turn == chess.BLACK else -1
        
        # if black wins return 1 if black, -1 if white
        elif result == "0-1":
            return 1 if self.board.turn == chess.WHITE else -1

        # if draw return 0
        return 0
    
    def piece_to_channel(self, piece):
        """
        convert a piece object to the appropriate channel index
        """
        # subtract 1 to convert to 0-indexed
        piece_type = piece.piece_type - 1

        # white pieces are 0-5, black pieces are 6-11
        return piece_type + (6 if piece.color == chess.BLACK else 0)
    
    # static method is a method that is bound to the class and not the object of the class
    @staticmethod
    def square_to_coords(square):
        """
        convert a square index to row, col coordinates
        """

        # row idx is 7 - square // 8, col idx is square % 8s
        return (7 - square // 8, square % 8)
    
class MoveConverter:
    """
    converts between chess.Move objects and indices
    """

    def __init__(self):
        self.move_indices = {}
        self.build_move_index

    def build_move_index(self):
        """
        builds a dictionary mapping chess.Move objects to indices
        """
        idx = 0
        # loop through the squares move comes from and goes to
        for from_sq in chess.SQUARES:
            for to_sq in chess.SQUARES:
                # add move to the dictionary at idx
                self.move_indices[chess.Move(from_sq, to_sq)] = idx
                idx += 1
    
    def move_to_idx(self, move):
        """
        given a move, convert it to an index
        """
        return self.move_indices.get(move)
    
    def idx_to_move(self, idx):
        """
        given an index, convert it to a move
        """
        
        # reverse mapping - loop through dictionary to find move with index
        for move, index in self.move_indices.items():
            if index == idx:
                return move
        return None