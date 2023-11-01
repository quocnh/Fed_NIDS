import torch
from torch import nn

import math

# Define the DNN model
# class DNN(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super().__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_size, num_classes)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         return x
#     @property
#     def device(self):
#         return next(self.parameters()).device

class DNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super().__init__()
        self.fc_layers = nn.ModuleList()
        layer_sizes = [input_size] + hidden_sizes
        for i in range(len(hidden_sizes)):
            self.fc_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            self.fc_layers.append(nn.ReLU())
        self.fc_layers.append(nn.Linear(hidden_sizes[-1], num_classes))

    def forward(self, x):
        for layer in self.fc_layers:
            x = layer(x)
        return x

    @property
    def device(self):
        return next(self.parameters()).device
