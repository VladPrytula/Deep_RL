import gym
from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70


class Net(nn.Module):
    """
    The output from the network is a probability distribution over actions,
    so a straightforward way to proceed would be to include softmax nonlinearity after the last layer.
    However, in the preceding network we don't apply softmax to increase the numerical stability of the 
    training process. Rather than calculating softmax (which uses exponentiation) and then calculating 
    cross-entropy loss (which uses logarithm of probabilities), 
    we'll use the PyTorch class, nn.CrossEntropyLoss, which combines both softmax and cross-entropy in a single, 
    more numerically stable expression. 
    CrossEntropyLoss requires raw, unnormalized values from the network (also called logits), 
    and the downside of this is that we need to remember to apply softmax every time we 
    need to get probabilities from our network's output
    """
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)


# let us create a pair naimed tuples
Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


def iterate_batches(env, net, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    sm = nn.Softmax(dim=1)

    #now let us run the environment loop
    while True:
        # nn.Module expects a batch of data, same is true for Networks
        # so we convert our 4D observation vector into a 1x4 tensor
        obs_v = torch.FloatTensor([obs])
        # since our network does not have a nonlinearity at the end
        # we have to apply softmax to get "probabilities"
        act_probs_v = sm(net(obs_v))
        # 
        act_probs = act_probs_v.data.numpy()[0]
        # sample action according to probs
        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done, _ = env.step(action)
        # pass this action to the environment to get our next observation, 
        # our reward, and the indication of the episode ending:
        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=obs, action=action))

        if is_done:
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()
            # we actually return the batch if episode was sufficiently large for processing
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs

        # ONE MIGHT NOTICE: that since training of the network and batch production
        # is done in one thread, so this might somewhat hit the performance

def filter_batch(batch, percentile):
    """
    This is the core of the cross entropy method:
    We want to filter those episode that bring no or little benefit to us even if the lenght
    of the episode was sufficient to be produced

    We use percentile to cut off

    Afterwards we average the avard for the episode TODO: this is for monitoroing only
    TODO: do not fully understand this
    """
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    # filter episodes
    train_obs = []
    train_act = []
    for sample in batch:
        if sample.reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, sample.steps))
        train_act.extend(map(lambda step: step.action, sample.steps))

    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)

    # reward_bound and reward_mean will be passed to tensorboard
    return train_obs_v, train_act_v, reward_bound, reward_mean


if __name__ == '__main__':
    env = gym.make("CartPole-v0")
    # if we want to record videos at each step
    # env = gym.wrappers.Monitor(env, directory="mon", force=True)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)
    writer = SummaryWriter()


    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()
        
        print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (
            iter_no, loss_v.item(), reward_m, reward_b))
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)
        if reward_m > 199:
            print("Solved!")
            break
    writer.close()