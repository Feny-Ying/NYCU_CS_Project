# code adapted from https://github.com/wendelinboehmer/dcg
from typing import Optional, Dict

import torch.nn as nn
import torch.nn.functional as F


class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        if self.args.use_rnn:
            self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        else:
            self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state, dormant_record: Optional[Dict]):

        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        q = self.fc2(h)

        if dormant_record is not None:
            # x.shape: [batch, hidden_dim]
            # h.shape: [batch, hidden_dim]
            # Compute mean of abs of x and h over batch dimension
            x_mean = x.detach().abs().mean(dim=0)
            h_mean = h.detach().abs().mean(dim=0)
            dormant_record['x'].append(x_mean.cpu().numpy())
            dormant_record['h'].append(h_mean.cpu().numpy())
        return q, h
