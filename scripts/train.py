import os, sys, glob
import random
import chess
import wandb
import numpy as np
from tqdm.auto import tqdm
from dotenv import load_dotenv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.append('../')

# custom imports
from utils.env import *
from utils.model import *

load_dotenv()
os.environ['WANDB_API_KEY'] = os.getenv('WANDB_API_KEY')

class ChessTrainer:

    def __init__(self):

        self.wins = 0
        self.draws = 0
        self.losses = 0

        self.model = ChessNet()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000
            )
        self.gamma = 0.99
        
        # device mgmt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def evaluate_vs_random(self, num_games=10):
        """
        Evaluate model against random agent
        """
        wins = 0
        draws = 0

        for _ in range(num_games):
            # play as both white and black
            for color in [chess.WHITE, chess.BLACK]:
                env = ChessEnv()
                state, mask = env.reset()
                done = False

                while not done:
                    if env.board.turn == color:
                        state_tensor = torch.tensor(state).unsqueeze(0)
                        policy_logits, _ = self.model(state_tensor)
                        action = torch.argmax(policy_logits).item()
                    else:
                        legal_moves = list(env.board.legal_moves)
                        move = random.choice(legal_moves)
                        try:
                            action = env.move_converter.move_to_idx(move)
                        except ValueError:
                            done = True # if illegal move, end game
                            reward = -1
                            break
                
                    state, mask, reward, done = env.step(action)
        
                # record results
                result = env.board.result()
                if result == "1-0":
                    wins += 1 if color == chess.WHITE else 0
                elif result == "0-1":
                    wins += 1 if color == chess.BLACK else 0
                else:
                    draws += 1
            
        total_games = num_games * 2
        win_rate = (wins + 0.5*draws) / total_games
        return win_rate
    
    def self_play(self, temperature=0.7):

        # init environment
        env = ChessEnv()

        # init buffers, state (start game)
        states, moves, values, rewards, masks = [], [], [], [], []
        state, mask = env.reset()

        # play game
        while True:

            # turn state into tensor
            state_tensor = torch.tensor(state, device=self.device).unsqueeze(0)

            # get policy and value from model
            policy_logits, value = self.model(state_tensor)

            # apply mask, softmax policy
            mask_tensor = torch.tensor(mask).bool()
            policy_logits = policy_logits.squeeze(0)
            policy_logits = torch.where(
                mask_tensor,
                policy_logits,
                torch.tensor(-1e8, device=policy_logits.device)
            )
            policy = F.softmax(policy_logits / temperature, dim=0)

            # store data
            states.append(state)
            masks.append(mask)

            # sample action
            action = torch.multinomial(policy, 1).item()
            moves.append(action)
            values.append(value.item())

            # take a step in the environment (make your move, get new state)
            state, mask, reward, done = env.step(action)
            rewards.append(reward)

            if done:
                break

        return states, moves, values, rewards, masks

    def train_step(self):

        # play a single game, calculate returns
        states, moves, values, rewards, masks = self.self_play()
        returns = self.calculate_returns(rewards)

        # convert to tensors
        states = torch.tensor(np.stack(states), dtype=torch.float32)
        moves = torch.tensor(moves, dtype=torch.long)
        returns = torch.tensor(returns, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)

        # forward pass
        policy_logits, values_pred = self.model(states)
        values_pred.squeeze()

        # policy loss
        mask = torch.stack([torch.tensor(m) for m in masks])
        policy_logits[mask == 0] = -float('inf')

        # policy loss = -log(policy logits) * expected returns (advantage) for each move
        # ACTOR
        advantages = returns - values_pred.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # normalize advantages s.t. they don't dominate
        
        log_probs = F.log_softmax(policy_logits, dim=-1)
        selected_log_probs = log_probs.gather(1, moves.unsqueeze(1))
        policy_loss = -torch.mean(selected_log_probs * advantages)


        # value loss
        # CRITIC
        value_loss = torch.mean((returns - values_pred) ** 2)

        # total loss 
        loss = policy_loss + value_loss

        # backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # clip gradients
        nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        
        avg_reward = np.mean(rewards.numpy())

        return loss.item(), value_loss.item(), policy_loss.item(), avg_reward
    
    def calculate_returns(self, rewards):
        returns = []
        R = 0
        for r in reversed(rewards):
            # return = reward + discount rate * prev. returns
            R = r + self.gamma * R
            returns.insert(0, R)
        return returns

if __name__ == '__main__':

    # init trainer
    trainer = ChessTrainer()
    win_rates = []
    win_rate = trainer.evaluate_vs_random(num_games=20)

    # init wandb for tracking
    wandb.init(project='chess-rl')

    # play 1000 games
    for i in range(1000):
        loss, value_loss, policy_loss, avg_reward = trainer.train_step()
        
        # store win rates for plotting
        if i % 100 == 0:
            current_win_rate = trainer.evaluate_vs_random(num_games=20)
            win_rates.append(win_rate)
            win_rate = current_win_rate


        print(f'\r\033[KIteration {i} | '
              f'Total Loss: {loss:.4f} | '
              f'Value Loss: {value_loss:.4f} | '
              f'Policy Loss: {policy_loss:.4f} | ' 
              f'Win Rate: {win_rate:.2%} | '
              f'Avg. Reward: {avg_reward:.4f}', end='', flush=True)
        wandb.log({'iteration': i, 'loss': loss, 'policy_loss': policy_loss, 'value_loss': value_loss, 'win_rate': win_rate, 'avg_reward': avg_reward})


    # save model
    torch.save(trainer.model.state_dict(), 'model.pth')
    wandb.finish()
