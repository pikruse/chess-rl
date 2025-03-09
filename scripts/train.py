import os, sys, glob
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
        self.model = ChessNet()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.gamma = 0.99
    
    def self_play(self):

        # init environment
        env = ChessEnv()

        # init buffers, state (start game)
        states, moves, values, rewards, masks = [], [], [], [], []
        state, mask = env.reset()

        # play game
        while True:

            # turn state into tensor
            state_tensor = torch.tensor(state).unsqueeze(0)

            # get policy and value from model
            policy_logits, value = self.model(state_tensor)

            # apply mask, softmax policy
            mask_tensor = torch.tensor(mask).bool()
            policy_logits = policy_logits.squeeze(0)
            policy_logits[~mask_tensor] = -float('inf') # set invalid moves to -inf
            policy = F.softmax(policy_logits, dim=0)

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
        states = torch.tensor(states, dtype=torch.float32)
        moves = torch.tensor(moves, dtype=torch.long)
        returns = torch.tensor(returns, dtype=torch.float32)

        # forward pass
        policy_logits, values_pred = self.model(states)
        values_pred.squeeze()

        # policy loss
        mask = torch.stack([torch.tensor(m) for m in masks])
        policy_logits[mask == 0] = -float('inf')

        # policy loss = -log(policy logits) * expected returns (advantage) for each move
        # ACTOR
        policy_loss = -torch.mean(
            F.log_softmax(policy_logits, dim=1)[
                torch.arange(len(moves)), moves
                ] * (returns - values_pred.detach())
        )

        # value loss
        # CRITIC
        value_loss = torch.mean((returns - values_pred ** 2))

        # total loss 
        loss = policy_loss + value_loss

        # backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), torch.mean(rewards).item()
    
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

    # init wandb for tracking
    wandb.init(project='chess-rl')

    # play 1000 games
    for i in range(1000):
        loss, avg_reward = trainer.train_step()
        print(f'Iteration {i} | Loss: {loss} | Avg. Reward', end='\r')
        wandb.log({'iteration': i, 'loss': loss, 'avg_reward': avg_reward})
    
    # save model
    torch.save(trainer.model.state_dict(), 'model.pth')
    wandb.finish()
