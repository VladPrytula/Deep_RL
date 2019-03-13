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

And let us now try to write our first agent, the simpliest agent possible - the one that takes random actions all the time

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
    # wrap our environment in a monitor
    env. = gym.wrappers.Monitor(env, "recording")
    #run the random agent
    run_random_agent(env)
~~~~
> Note: there is a certain issue that one might run into while trying to record the episode that. On linux that can be solved using the Xvbf
> `xvbf-run -s "--screen 0 640x480x24" python random_agent_cartpole.py`

#### Cross entoropy method for CartPole invironment
References 

[ P. T. de Boer, D. P. Kroese, S. Mannor, and R. Y. Rubinstein. A tutorial on the cross-entropy method. Annals of Operations Research, 134(1):19–67, 2005.](https://people.smp.uq.edu.au/DirkKroese/ps/aortut.pdf)

[Dirk P. Cross-Entropy method. Kroese School of Mathematics and Physics](https://people.smp.uq.edu.au/DirkKroese/ps/eormsCE.pdf)

The CE method involves an iterative procedure where each iteration can be broken
down into two phases:
1. Generate a random data sample (trajectories, vectors, etc.) according to a specified mechanism.
2. Update the parameters of the random mechanism based on the data to produce a “better” sample in the next iteration.


> The following is mostly from here: [The Cross-Entropy Method for Estimation Dirk P. Kroese1, Reuven Y. Rubinstein2, and Peter W. Glynn](https://web.stanford.edu/~glynn/papers/2013/KroeseRubinsteinG13.pdf)

As it is stated in the above referenece article : _"The CE method was introduced by Rubinstein (1999, 2001), extending earlier
work on variance minimization (Rubinstein, 1997). Originally, the CE method
was developed as a means of computing rare-event probabilities; that is, very
small probabilities—say less than 10−4. Naive Monte Carlo estimation of such a probability requires a large simulation effort, inversely proportional to the
magnitude of the rare-event probability. The CE method is based on two ideas.
The first idea is to estimate the probability of interest by gradually changing the
sampling distribution, from the original to a distribution for which the rare event is
much more likely to happen. To remove the estimation bias, importance sampling is
used. The second idea is to use the CE distance to construct the sequence of sampling
distributions. This significantly simplifies the numerical computation at each step,
and provides fast and efficient algorithms that are easy to implement by practitioners"_

#### Problem setting:
Genearlly we want to get the estimation of the expectation
$$
l = \mathbb{E}_f[H(X)] = \int H(x) f(x) dx,
$$
where $H$ is some real-valued function and $f$ is the probability density fucntion of a random variable $X$

In the RL setting $H(x)$ is a reward value obtained by some policy $x$ (**TODO: define what policy is**) and $f(x)$ is a distribution of all possible policies. We don't want to maximize our reward by searching all possible policies, instead we want to find a way to approximate $f(x)H(x)$ by some $q(x)$, iteratively minimizing the distance between them.

*Definition: Policy*:

In our case of the CartPole the $H(x)$ can be replaced by an indicator function when the total reward for the episode is higher than certain threshold.**TODO: why can we do it?** 