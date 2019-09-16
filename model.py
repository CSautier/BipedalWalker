import torch
import torch.nn as nn
import torch.nn.functional as F
from parameters import parameters
from torch.distributions import Normal


class MLP(nn.Module):
    def __init__(self, obs_space, action_space):
        super(MLP, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(obs_space, obs_space * 8),
            nn.LeakyReLU(0.1),
            nn.Linear(obs_space * 8, obs_space * 8),
            nn.LeakyReLU(0.1),
            nn.Linear(obs_space * 8, 1)
        )

        self.mean = nn.Sequential(
            nn.Linear(obs_space, obs_space * 8),
            nn.LeakyReLU(0.1),
            nn.Linear(obs_space * 8, obs_space * 8),
            nn.LeakyReLU(0.1),
            nn.Linear(obs_space * 8, action_space),
            nn.Tanh()
        )
        self.logstd = nn.Parameter(torch.Tensor([0, 0, 0, 0]))

    def forward(self, x):
        mean = self.mean(x)
        actor = Normal(mean, torch.exp(self.logstd))
        if self.training:
            critic = self.critic(x)
            return actor, critic
        return actor

    def loss(self, observations, rewards, actions, old_prob):
        prob_distribution, reward_predicted = self.forward(observations)
        # prob = torch.prod(prob_distribution.cdf(actions), dim=1)
        r = (torch.prod(prob_distribution.cdf(actions), dim=1) + 1e-10) / (old_prob + 1e-10)
        advantage = (rewards - reward_predicted).detach().squeeze()
        # print(mean)
        lossactor = - parameters.ACTOR_COEFF \
                    * torch.mean(torch.min(r * advantage,
                                           torch.clamp(r,
                                                       min=(1. - parameters.LOSS_CLIPPING),
                                                       max=(1. + parameters.LOSS_CLIPPING))
                                           * advantage))
        losscritic = F.mse_loss(reward_predicted, rewards)
        return lossactor, losscritic
