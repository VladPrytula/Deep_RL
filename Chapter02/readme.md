![](images/logo5.png)

![](https://img.shields.io/badge/Uploaded-100%25-green.svg)  

### The first steps of Reinforcement learning journey (from Cartpole to Doom). 

# Section 1. Cartpole

### Prerequisites
* Python and PyTorch
* basic lienear algebra

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
        pass

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

    def reset(self):
        self.time_to_live = 10
        self.is_done = False 
~~~~