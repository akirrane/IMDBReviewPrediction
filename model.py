# Import Neural Network module from torch
import torch.nn as nn

# Define our own Neural Network class based off of torch
class NeuralNetwork(nn.Module):
    # Constructor, initialize layers and activation function
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) # Input layer
        self.l2 = nn.Linear(hidden_size, hidden_size) # Hidden layer 1
        self.l3 = nn.Linear(hidden_size, output_size) # Hidden layer 2
        self.relu = nn.ReLU() # Set activation function as ReLU

    # Feedforward function, calculates activations for each layer and returns final output
    def forward(self, x):
        out = self.l1(x) # Layer 1 output
        out = self.relu(out) # Activated layer 1 ouput
        out = self.l2(out) # Layer 2 output
        out = self.relu(out) # Activated layer 1 ouput
        out = self.l3(out) # Layer 3 output
        return out