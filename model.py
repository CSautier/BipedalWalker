import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

ENTROPY_COEFF = 1e-4
ACTOR_COEFF = 0.1
CENTERING_COEFF = 0.0001
LOSS_CLIPPING = 0.2


class MLP(nn.Module):
    def __init__(self, obs_space, action_space):
        super(MLP, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(obs_space, obs_space * 3),
            nn.Tanh(),
            nn.Linear(obs_space * 3, obs_space * 4),
            nn.Tanh(),
            nn.Linear(obs_space * 4, 1)
        )

        self.mean = nn.Sequential(
            nn.Linear(obs_space, obs_space * 3),
            nn.Tanh(),
            nn.Linear(obs_space * 3, obs_space * 4),
            nn.Tanh(),
            nn.Linear(obs_space * 4, action_space),
            nn.Tanh()
        )
        self.logstd = nn.Parameter(torch.Tensor([0,0,0,0]))

    def forward(self, x):
        mean = self.mean(x)
        actor = Normal(mean, torch.exp(2*self.logstd))
        if self.training:
            critic = self.critic(x)
            return actor, critic
        return actor

    def _loss(self, observations, rewards, actions, old_prob):
        prob_distribution, reward_predicted = self.forward(observations)
        prob = torch.prod(prob_distribution.cdf(actions), dim=1)
        old_prob = old_prob.squeeze()
        r = (prob + 1e-10) / (old_prob + 1e-10)
        advantage = (rewards - reward_predicted).detach().squeeze()
        mean = prob_distribution.mean
        # print(mean)
        losscenter = CENTERING_COEFF * torch.mean(-1 - 1/((mean-1)*(mean+1)-1e-8))  # CENTERING_COEFF * torch.mean(mean*mean)
        lossactor = -ACTOR_COEFF*torch.mean(torch.min(r * advantage,
                                                      torch.clamp(r, min=(1.-LOSS_CLIPPING), max=(1.+LOSS_CLIPPING))
                                                      * advantage))
        losscritic = 5*F.mse_loss(reward_predicted, rewards)
        print('Loss actor: {0:7.3f}  Loss critic: {1:7.3f}  Loss center: {2:6.3f}  Std: {3}'.format(
            1000*lossactor, 1000*losscritic, 1000*losscenter, torch.exp(2*self.logstd).data.cpu().numpy()), flush=True)
        return lossactor + losscritic + losscenter  # + lossentropy


# class BTA(nn.Module):
#     def __init__(self, obs_space, action_space):
#         super(BTA, self).__init__()
#         self.critic = nn.Sequential(
#             nn.Linear(obs_space, obs_space * 3),
#             nn.Tanh(),
#             nn.Linear(obs_space * 3, obs_space * 4),
#             nn.Tanh(),
#             nn.Linear(obs_space * 4, 1),
#             nn.Tanh()
#         )
#
#         self.mean = nn.Sequential(
#             nn.Linear(obs_space, obs_space * 3),
#             nn.Tanh(),
#             nn.Linear(obs_space * 3, obs_space * 4),
#             nn.Tanh(),
#             nn.Linear(obs_space * 4, action_space),
#             nn.Tanh()
#         )
#         self.logstd = nn.Parameter(torch.Tensor([-1.5]))
#
#     def forward(self, x):
#         mean = self.mean(x)
#         logstd = self.logstd
#         std = torch.exp(logstd)
#         action = torch.normal(mean, std)
#         #        actor = Normal(mean, std)
#         if self.training:
#             critic = self.critic(x)
#             return mean, logstd, critic
#         return action
#
#     def _loss(self, observations, rewards, actions):
#         mean, logstd, reward_predicted = self.forward(observations)
#         print(mean)
#         rewards_mean = torch.mean(rewards)
#         print(rewards_mean)
#         prob_distribution = Normal(mean, torch.exp(logstd))
#         prob = torch.sum(prob_distribution.cdf(actions), dim=1)
#         old_prob = prob.detach()
#         r = (prob + 1e-10) / (old_prob + 1e-10)
#         bta = (1. * (rewards>rewards_mean) - 1. * (rewards<rewards_mean)).detach()
#         print(bta)
#         # lossentropy = -torch.mean(ENTROPY_COEFF * prob_distribution.entropy())
#         advantage = (bta*nn.MSELoss(reduction='none')(bta, reward_predicted)).squeeze()
#         print(advantage)
#         print(advantage.size())
#         losscenter = CENTERING_COEFF * torch.mean(mean*mean)
#         lossactor = -ACTOR_COEFF * torch.mean(r * advantage)
#         losscritic = F.mse_loss(reward_predicted, bta)
#         print('Loss actor: {0:7.3f}  Loss critic: {1:7.3f}  Loss center: {2:6.3f}  logstd: {3:7.3f}'.format(
#             1000*lossactor, 1000*losscritic, 1000*losscenter, logstd.item()), flush=True)
#         return
#         return lossactor + losscritic + losscenter  # + lossentropy