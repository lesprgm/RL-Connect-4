import torch
import torch.nn as nn
import torch.nn.functional as F

class connect4SelfPlayModel(nn.Module):
    def __init__(self, input_size=42, H1_size=256, H2_size=128, output_size=7):
        super(connect4SelfPlayModel, self).__init__()
        self.fc1 = nn.Linear(input_size, H1_size)
        self.fc2 = nn.Linear(H1_size, H2_size)
        self.fc3 = nn.Linear(H2_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x