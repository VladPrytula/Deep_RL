import gym

import collections
from tensorboardX import SummaryWriter

import numpy as np

ENV_NAME = "FrozenLake-v0"
GAMMA = 0.9
TEST_EPISODES = 20


"""
. Reward table: 
    A dictionary with the composite key "source state" + "action" + "target state". 
    The value is obtained from the immediate reward.

. Transitions table: 
    A dictionary keeping counters of the experienced transitions. 
    The key is the composite "state" + "action" and the value is another dictionary that maps 
    the target state into a count of times that we've seen it. 
    For example, if in state 0 we execute action 1 ten times, after three times it leads us to 
    state 4 and after seven times to state 5. 
    Entry with the key (0, 1) in this table will be a dict {4: 3, 5: 7}. 
    We use this table to estimate the probabilities of our transitions.

. Value table: 
    A dictionary that maps a state into the calculated value of this state.
"""

"""
    Then we define the Agent class, which will keep our tables and contain 
    functions we'll be using in the training loop
"""

class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()

        # this maps (state, action, target_state) to an obtained reward
        self.rewards = collections.defaultdict(np.float32)

        # this collections.Counter thing has to be explained
        # transits is mapping from the (state,action) -> {st1:count, st2:count}
        self.transits = collections.defaultdict(collections.Counter)

        # maps st -> calculated value of this state V(st1)
        # TODO: should I initialize it to something?
        self.values = collections.defaultdict(np.float32)

    def play_n_random_steps(self, count):
        """
        we gather random experiences from the environment
        and update the reward and transition table
        We do not wait for the episode to end if it lasts more than N steps

        This is in contrary to Cross Entropy and value iteration
        that should wait for the episode to end
        """
        for _ in range(count):
            action = self.env.action_space.sample()
            new_state, reward, is_done, _ = self.env.step(action)
            self.rewards[(self.state, action, new_state)] = reward
            self.transits[(self.state, action)][new_state] += 1
            self.state = self.env.reset() if is_done else new_state

    def calc_action_value(self, state, action):
        # so what are the experemetnal counts for transitions to new states
        # from current state under current action 
        # we must understand we might have never seen all the possible states
        target_counts = self.transits[(state, action)]
        
        # get the normalization value
        total = sum(target_counts.values())

        action_value = 0.0

        for tgt_state, count in target_counts.items():
            reward = self.rewards[(state, action, tgt_state)]

            # the funny thing here is that we assume that self.values are given
            # TODO: I am not really sure about that
            action_value += (count / total)*(reward + GAMMA * self.values[tgt_state])

        return action_value

    def select_action(self, state):
        """
        Straightforward best action selection using
        the cacl_action_value(state, action) function!
        """
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.calc_action_value(state, action)
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action


    def play_episode(self, env):
        """
        plays one full (!) episode using the select_action(state) function
        while we play we also collect rewards, transits and compute total_reward

        we also have to be carefull not to mess with the state of the env
        that is being used to gather random data
        """
        total_reward = 0.0
        state = env.reset()
        while True:
            action = self.select_action(state)
            new_state, reward, is_done, _ = env.step(action)
            # here it should be not that direct actually
            # since reward might be also stochastic?
            self.rewards[(state, action, new_state)] = reward
            self.transits[(state, action)][new_state] += 1
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward

    
    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            state_values = [self.calc_action_value(state, action) for 
                            action in range(self.env.action_space.n)]
            self.values[state] = max(state_values)




if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter()

    iter_no = 0
    best_reward = 0.0
    writer.add_scalar("reward", 0.0, iter_no)

    while True:
        iter_no += 1

        # generate n random steps and fill our transtion tables
        # use value_iterations over all states (TODO: so we know all of them?)
        agent.play_n_random_steps(100)
        agent.value_iteration()

        # now let use the tables to evaluate
        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
        print(reward)
        print(iter_no)
        writer.add_scalar("reward", reward, iter_no)
        print
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward
            if reward > 0.8:
                print("Solved in % iterations" % iter_no)
                break

    writer.close()