import gym
import collections
from tensorboardX import SummaryWriter
import numpy as np

ENV_NAME = "FrozenLake-v0"
GAMMA = 0.9
ALPHA = 0.2
TEST_EPISODES = 20

class Agent:
    def __init__(self):
        """
        in tabular Q-learning we are not processing all possible states and action, but
        depend on the ones that we sample from the environment

        By interacting with the environment we obtain the tuple (s,a,r,s')
        and decide which action we have to take with help of some version of Exploration versus Esploitation

        Update the Q(s,a) using the Bellman approximation
        Q_{s,a} \leftarrow r + \gamma max_{a' \in A} Q_{s',a'}

        the question here why do we need the history of rewards and and transition counters?
        Answers is since we are actually not trying to estimate the transition probabilities???
        but instead we are using the convex envelope for the Bellman update
        """
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        # mpas (state, action) to reward
        self.action_values = collections.defaultdict(np.float32)

    def sample_env(self):
        """
        we use this method to obtain next transition from the environment
        """
        action = self.env.action_space.sample()
        old_state = self.state
        new_state, reward, is_done, _ = self.env.step(action)
        self.state = self.env.reset() if is_done else new_state
        return (old_state, action, reward, new_state)

    def best_value_and_action(self, state):
        best_value, best_action = None, None
        for action in range(self.env.action_space.n):
            # it is important to use [] here and not get!!! with a default value of 0.0
            action_value = self.action_values[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_value, best_action

        #print('best action is {}'.format(best_action))
        #print('best value is {}'.format(best_value))
        return best_value, best_action


    def value_update(self, state, action, reward, next_state, iter_no):
        """
        The most important part of the algorithm:
        We update the value table using one step from the environment
        """
        best_value, _ = self.best_value_and_action(next_state)
        new_value = reward + GAMMA * best_value
        old_value = self.action_values[(state, action)]
        # the thing is not very stable actually 
        # and if the learning rate is constant if frequently overshoots
        if iter_no > 5000:
            self.action_values[(state, action)] = (1-ALPHA/4) * old_value + (ALPHA/4) * new_value
        else:
            self.action_values[(state, action)] = (1-ALPHA) * old_value + (ALPHA) * new_value
    def play_episode(self, env):
        total_reward = 0.0
        state = env.reset()
        while True:
            _, action = self.best_value_and_action(state)
            new_state, reward, is_done, _ = env.step(action)
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward


if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-q-learning")

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        s, a, r, next_s = agent.sample_env()
        agent.value_update(s, a, r, next_s, iter_no)

        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward
        if reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            break
    writer.close()