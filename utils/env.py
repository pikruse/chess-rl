import chess
import numpy as np

class ChessEnv:
    """
    environment for with board state representation + move masking
    """

    def __init__(self):
        self.board = chess.Board()
        self.move_converter = MoveConverter()
        self.action_space = 4672 # 8 x 8 x 73 = 4672 possible moves

    def get_mask(self):
        """
        generate mask for all 4672 possible moves
        """

        # init mask
        mask = np.zeros(self.action_space, dtype=np.float32) # 8 x 8 x 73 = 4672 (73 possible moves per square)

        # loop through legal moves to fill in mask
        for move in self.board.legal_moves:
            try:
                # get index of move and set mask to 1 at idx
                idx = self.move_converter.move_to_idx(move)
                mask[idx] = 1
            except ValueError:
                continue
        return mask
    
    def step(self, action):
        """
        Takes a step in the environment (makes a move)
        """

        # get move from action
        move = self.move_converter.idx_to_move(action, self.board)

        # if move is illegal, return state, mask, -1 reward
        if not move or move not in self.board.legal_moves:
            return self.get_state(), self.get_mask(), -1, True
        
        # make move, init reward, check if game is over
        self.board.push(move)
        reward = 0
        done = self.board.is_game_over()

        # if game is over, get result and reward
        if done:
            reward = 1 if self.board.result() == "1-0" else -1
        
        # add material balance reward
        piece_values = {1:1, 2:3, 3:3, 4:5, 5:9, 6:0}
        
        # get material balance
        white_material = sum(piece_values[abs(p.piece_type)]
                             for p in self.board.piece_map().values()
                             if p.color == chess.WHITE)
        black_material = sum(piece_values[abs(p.piece_type)]
                             for p in self.board.piece_map().values()
                             if p.color == chess.BLACK)
        
        # scale and add to reward
        max_material = 39  # sum of all piece values * 2
        material_diff = (white_material - black_material) / max_material
        reward += 0.5 * material_diff  # Increased from 0.1 to 0.5
        reward += 0.2 * (len(self.board.move_stack) % 2)  # Bonus for making progress

        # return state, mask, reward, done
        return self.get_state(), self.get_mask(), reward, done
    
    def reset(self):
        self.board = chess.Board()
        return self.get_state(), self.get_mask()
    
    def get_state(self):
        """
        gets a 14 x 8 x 8 board state representation (14 channels for pieces + "special" moves, 8x8 board)
        """

        # init state representation
        state = np.zeros((15, 8, 8), dtype=np.float32)
        
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

    def calculate_reward(self, result):
        """
        given a game result, calculate the reward
        """

        result = self.board.result()
        # Fixed:
        if result == "1-0":
            return 1  # White won
        elif result == "0-1":
            return -1  # Black won
        else:
            return 0 # Draw
    
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
        """
        Create move planes and dictionaries for move conversion
        """
        self.move_lookup = self._create_move_lookup()
        self.idx_to_move_mapping = {}

    def _create_move_lookup(self):
        """
        Create full 4672 move mapping using alphazero-style encoding
        """
        move_lookup = {}
        idx = 0

        # queen-style moves (cover all pieces except knight)
        directions = [(1,0), (-1,0), (0,1), (0,-1),
                      (1,1), (1,-1), (-1, 1), (-1,-1)]
        for dx, dy in directions:
            for distance in range(1, 8):
                move_lookup[idx] = (('queen', dx, dy, distance))
                idx += 1
        
        # knight moves
        knight_moves = [(2,1), (2,-1), (-2,1), (-2,-1),
                        (1,2), (1,-2), (-1,2), (-1,-2)]
        for dx, dy in knight_moves:
            move_lookup[idx] = (('knight', dx, dy))
            idx += 1

        # underpromotions
        promotions = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
        for promo in promotions:
            for dx in [-1, 0, 1]:
                move_lookup[idx] = (('underpromotion', promo, dx))
                idx += 1
        
        return move_lookup
    
    def move_to_idx(self, move):
        """
        convert a chess.Move object to an index

        Args:
            move: chess.Move object
        Returns:
            index: int
        """

        from_sq = move.from_square
        to_sq = move.to_square
        promo = move.promotion
        from_rank, from_file = divmod(from_sq, 8)
        to_rank, to_file = divmod(to_sq, 8)

        dx = to_file - from_file
        dy = to_rank - from_rank
        promo = move.promotion

        # calculate base index for this square
        square_base = from_sq * 73

        # queen-style moves (0-55)
        if not promo or promo == chess.QUEEN:
            direction_map = { (-1,0): 0, (1,0): 1, (0,-1): 2, (0,1): 3,
                              (-1,-1): 4, (-1,1): 5, (1,-1): 6, (1,1): 7 }
            if (dx, dy) in direction_map:
                dir_idx = direction_map[(dx, dy)]
                distance = max(abs(dx), abs(dy))
                if 1 <= distance <= 7:
                    return square_base + dir_idx * 7 + distance - 1
        
            # knight moves (56-63)
            knight_moves = [
                (-2, -1), (-2, 1), (-1, -2), (-1, 2),
                (1, -2), (1, 2), (2, -1), (2, 1)
            ]
            if (dx, dy) in knight_moves:
                knight_idx = knight_moves.index((dx, dy))
                return square_base + 56 + knight_idx
            
        # underpromotions (64-72)
        if promo and promo != chess.QUEEN:
            promo_map = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}
            dx_map = {-1: 0, 0: 1, 1: 2}
            promo_idx = promo_map[promo] * 3 + dx_map.get(dx, 1)
            return square_base + 64 + promo_idx
        
        # if no math found
        raise ValueError(f"Invalid move: {move}")
        

    def idx_to_move(self, idx, board):
        """
        convert action to chess.Move
        """
        
        if idx < 0 or idx >= 4672:
            raise ValueError(f"Invalid move index: {idx}")
        
        from_sq = idx // 73
        move_type_idx = idx % 73
        from_file = from_sq % 8
        from_rank = from_sq // 8

        # queen moves (0-55)
        if move_type_idx < 56:
            dir_idx = move_type_idx // 7
            distance = (move_type_idx % 7) + 1
            directions = [
                (-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)
            ]
            dx, dy = directions[dir_idx]
            dx *= distance
            dy *= distance

        # knight moves: (56-63)
        elif move_type_idx < 64:
            knight_moves = [
                (-2, -1), (-2, 1), (-1, -2), (-1, 2),
                (1, -2), (1, 2), (2, -1), (2, 1)
            ]
            dx, dy = knight_moves[move_type_idx - 56]
        
        # underpromotions (64-72)
        else:
            promo_idx = (move_type_idx - 64) // 3
            dx = (move_type_idx - 64) % 3 - 1
            promotions = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
            promo = promotions[promo_idx]

            # determine promo square from color
            if from_rank == 6:
                to_rank = 7
            elif from_rank == 1:
                to_rank = 0
            else:
                return None # invalid promotion square
            
            to_file = from_file + dx
            if 0 <= to_file < 8:
                to_sq = chess.square(to_file, to_rank)
                return chess.Move(from_sq, to_sq, promotion=promo)
            
            return None
        
        # regular move calculation
        to_file = from_file + dx
        to_rank = from_rank + dy
        if 0 <= to_file < 8 and 0 <= to_rank < 8:
            to_sq = chess.square(to_file, to_rank)
            return chess.Move(from_sq, to_sq)
        
        return None
        