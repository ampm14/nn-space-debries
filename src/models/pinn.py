import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.model = self.build_network(layers)

    def build_network(self, layers):
        net = []
        for i in range(len(layers) - 1):
            net.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                net.append(nn.Tanh())
        return nn.Sequential(*net)

    def forward(self, x):
        return self.model(x)
