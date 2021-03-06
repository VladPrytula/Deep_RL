import torch
import torch.nn as nn
import numpy as np


class DQN(nn.Module):
    """
    The output of the model is Q-values for every action available in the environment, 
    without nonlinearity applied (as Q-values can have any value). 
    The approach to have all Q-values calculated with one pass through the network helps us to increase speed 
    significantly in comparison to treating Q(s, a) literally and feeding observations and actions to the network 
    to obtain the value of the action
    """
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        print('init dqn')
        print(input_shape)

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        # print(' conv out size is {}'.format(conv_out_size))
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        # print('enterign _get_conv_out')
        out = self.conv(torch.zeros(1, *shape))
        # print('out is computed')
        return (int(np.prod(out.size())))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)
