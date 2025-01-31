import torch
from torch import nn
from torch.nn import functional as F


class CriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, agent_idx=-1, ma_critic=False, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._agent_idx = agent_idx
        self._ma_critic = ma_critic
        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action):
        if self._agent_idx != -1:
            if state.ndim == 3:
                state = state[:, self._agent_idx, :]
        if action.ndim == 3:
            if not self._ma_critic:
                action = action[:, self._agent_idx, :]
            else:
                action = torch.squeeze(action)
        state_action = torch.cat((state.float(), action.float()), dim=-1)
        features1 = F.relu(self._h1(state_action))
        features2 = F.relu(self._h2(features1))
        q = self._h3(features2)

        return torch.squeeze(q)


class MADDPGCriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h2.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h3.weight, gain=nn.init.calculate_gain("linear"))

    def forward(self, state, actions):
        """
        state: tensor of shape ((time), batch_size, n_features)
        actions: list of tensors of shape ((time), batch_size, n_actions)
        """
        state_action = torch.cat((state, actions), dim=-1)
        features1 = F.relu(self._h1(state_action))
        features2 = F.relu(self._h2(features1))
        q = self._h3(features2)
        return torch.squeeze(q)


class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, agent_idx=-1, **kwargs):
        super(ActorNetwork, self).__init__()

        self._n_input = input_shape[-1]
        self._n_output = output_shape[0]

        self._agent_idx = agent_idx
        self._h1 = nn.Linear(self._n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, self._n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('sigmoid'))

    def forward(self, state):
        if self._agent_idx != -1:
            if state.ndim == 3:
                state = state[:, self._agent_idx, :]
        if self._n_input != 1:
            if state.ndim >= 2:
                state = torch.squeeze(state, 1)
        features1 = F.relu(self._h1(state.float()))
        features2 = F.relu(self._h2(features1))
        a = F.sigmoid(self._h3(features2))
        # a = a * (self.mdp.info.action_space.high - self.mdp.info.action_space.low) + self.mdp.info.action_space.low

        return a
