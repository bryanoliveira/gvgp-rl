import torch.nn as nn
import torch

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Agent:
    def __init__(self, args):
        self.model = nn.Sequential(
            nn.Conv2d(args.history_length, 32, 8, stride=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),  #  colocar x = x.view(-1, 3136)
            nn.ReLU(),
            Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 6),
            nn.Softmax(1)
        )