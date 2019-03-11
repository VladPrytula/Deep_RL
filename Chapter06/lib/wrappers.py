import cv2
import gym
import gym.spaces
import numpy as np
import collections


# this is from https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
class FireResetEnv(gym.Wrapper):
    """
    Some of the envs require pressing fire at the start
    For example for pong we have to tell the agent to fire
    since while we can wait while it will acidentally learn to fire
    we will not really win a lot and loose a lot of time here
    """

    def __init__(self, env=None):
        # Take action on reset for environments that are fixed until firing
        super(FireResetEnv, self).__init__(env)

    def step(self, action):
        return self.env.step.action()

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        # self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        # for i in range(self._skip):
        #     obs, reward, done, info = self.env.step(action)
        #     if i == self._skip - 2: self._obs_buffer[0] = obs
        #     if i == self._skip - 1: self._obs_buffer[1] = obs
        #     total_reward += reward
        #     if done:
        #         break
        # make the loop more specific
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        # # Note that the observation on the done=True frame
        # doesn't matter
        # max_frame = self._obs_buffer.max(axis=0)
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        return max_frame, total_reward, done, info

    # def reset(self, **kwargs):
    #     return self.env.reset(**kwargs)
    def _reset(self):
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


# we will be using the 84x84 grayscale images.
class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            imp = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution"

        # we want to convert to a color that takes into accoutn that
        # human eye is not uniform across colors
        # http://www.tannerhelland.com/3643/grayscale-image-algorithm-vb6/
        # TODO: does it help somehow?????
        img = img[:, :, 0] * 0.3 + img[:, :, 1] * 59 + imp[:, :, 2] * 0.11
        resized_screen = cv2.resize(
            img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)

    def observation(self, observation):
        return ProcessFrame84.process(observation)


class BufferWrapper(gym.ObservationWrapper):
    # TODO: this implementation is not exactly clear one
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(
            self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(
            old_shape[-1], old_shape[0], old_shape[1]), dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


def make_env(env_name):
    env = gym.make(env_name)
    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = BufferWrapper(env, 4)

