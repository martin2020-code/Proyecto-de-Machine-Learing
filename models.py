import torch
import torch.nn as nn

class Sine(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, X):
        return torch.sin(X)
    

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.sine_stack = nn.Sequential(
            nn.Linear(2,100),
            Sine(),
            nn.Linear(100,100),
            Sine(),
            nn.Linear(100,100),
            Sine(),
            nn.Linear(100,3)
        )
    def forward(self, X):
        return self.sine_stack(X)