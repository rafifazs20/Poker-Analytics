import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepPokerNetwork(nn.Module):
    def __init__(self, input_size=106, hidden_sizes=[256, 256, 128], num_actions=5):
        super(DeepPokerNetwork, self).__init__()
        
        # Shared Layers
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        
        # Action space: 0=Fold, 1=Call, 2=Raise 0.5x Pot, 3=Raise 1x Pot, 4=Raise 3x Pot (Overbet)
        self.policy_head = nn.Linear(hidden_sizes[2], num_actions)
        self.value_head = nn.Linear(hidden_sizes[2], 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        policy_probs = F.softmax(self.policy_head(x), dim=-1)
        expected_value = torch.tanh(self.value_head(x))
        
        return policy_probs, expected_value