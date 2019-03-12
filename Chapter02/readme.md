![](https://cwiki.apache.org/confluence/download/attachments/69406797/inProgress.gif?version=1&modificationDate=1493416081000&api=v2)


### The first steps of Reinforcement learning journey (from Cartpole to Doom). 

# Section 1. Cartpole

### Prerequisites
* Python and PyTorch
* Open AI gym
* openCV

As a first excersise we will try to solve the "hello world" of RL problems, namely the "cartPole" problem. The original description is here [OpenAI Gym Cartpole](https://gym.openai.com/envs/CartPole-v0/).

From the RL point of view the two main entities are:

* Agent
* * A entity that takes an action (i.e. the role is active). In our case that will be some piece of code that descides what action to take based on the _observation_ obtained from the environment
* Environment
* * Certain representation of the world, which provide agent the observations and reward (can be heavily delayed in time). Environment can be deterministic (with respect to (state,action) pair) or stochastic.

## Toy example of the Environment and the Agent in Python
~~~~python
class Agent():
    def __init__(self):
        self.total_reward = 0.0

    def perform_action(self, env):
        actions = env.get_actions()
        action = random.choice(actions)
        reward, observatoin, is_done = env.step(action)
        self.total_reward += reward


class Environment():
    def __init__(self):
        self.time_to_live = 10
        self.is_done = False
    
    def step(self, action):
        reward, observation, is_done = None, None, False
        if self.is_done:
            reward = 1
            observation = None
            is_done = True
        else:
            reward = 1
            observation = [0,1, 0.2]
            self.time_to_live -= 1

        return reward, observation, is_done
    
    def is_done(self):
        return self.time_to_live == 0

    def get_actions(self, state):
        return [1, -1]

    def reset(self):
        self.time_to_live = 10
        self.is_done = False 
~~~~

Now let us look at a real example that is provided by OpenAI gym - CartPole

~~~~python
import gym
env = gym.make('CartPole-v0)
env.reset()
env.action_space()
env.obsrvation_space()
env.step(env.action_sapce.sample())
~~~~

And let us now try to write our first agent, the sipliest agent possible - the one that takes random actions all the time

~~~~python
# random agent can be found in random_agent_cartpole.py
import gym

def run_random_agent(env):
    total_reward: float = 0.0
    total_steps: int = 0
    obs = env.reset() # start the episode

    while True:
        sample_action = env.action_space.sample()
        obs, reward, is_done, info = env.step(sample_action)
        total_reward += reward
        total_steps += 1

        if is_done:
            break

    print("Episode contained %i steps, reward obtained is %.2f" % (total_steps, reward))

    pass

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    run_random_agent(env)
~~~~
