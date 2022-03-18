from operator import imod
from torch import nn


class MLP(nn.Module):
    def __init__(self, hidden_num=100):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, hidden_num),
            nn.ReLU(),
            nn.Linear(hidden_num, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
