import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim=2):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64,64)
        self.out = nn.Linear(64,output_dim)

    def forward(self,x):
        x = F.relu(self.fc1(x)) #relu application after first layer
        x = F.relu(self.fc2(x))
        return self.out(x) #final vals - q-values for each action
        