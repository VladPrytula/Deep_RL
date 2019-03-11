#!/usr/bin/env python3
from lib import wrappers
from lib import dqn_model

import argparse
import time
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter


DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19.5

GAMMA = 0.99  # Our gamma value used for Bellman approximation
BATCH_SIZE = 32
REPLAY_SIZE = 10000  # The maximum capacity of the buffer (REPLAY_SIZE)
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
# The count of frames we wait for before starting training to populate the replay buffer
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 10**5
EPSILON_START = 1.0
EPSILON_FINAL = 0.02


"""
defines our experience replay buffer, the purpose of which is to keep the last
transitions obtained from the environment (tuples of the observation, action, reward, done flag,
and the next state). Each time we do a step in the environment, we push the transition into
the buffer, keeping only a fixed number of steps, in our case 10k transitions.
For training, we randomly sample the batch of transitions from the replay buffer,
which allows us to break the correlation between subsequent steps in the environment.
"""
####
Experience = collections.namedtuple('Experience', field_names=[
                                    'state', 'action', 'reward', 'done', 'new_state'])
####


class ExperienceBuffer:
    """
    it basically exploits the capability of the deque class to maintain the given 
    number of entries in the buffer. In the sample() method, 
    we create a list of random indices and then repack the sampled entries into 
    NumPy arrays for more convenient loss calculation
    """
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(
            *[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
                np.array(dones, dtype=np.uint8), np.array(next_states)


class Agent:
    """
    Agent, which interacts with the environment and saves the result of the interaction 
    into the experience replay buffer (see above):
    """
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = env.reset()
        self.total_reward = 0.0


    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        new_state, reward, is_done, _ = self.env(action)
        self.total_reward += reward
        new_state = new_state

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)

        if is_done:
            done_reward = self.total_reward
            self._reset()

        return done_reward ## TODO: why do we return only done reward_
        
