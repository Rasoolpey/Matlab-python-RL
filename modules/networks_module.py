import torch
import torch.nn as nn
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x)) * self.max_action
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state, action):
        x = torch.relu(self.fc1(torch.cat([state, action], dim=1)))
        x = torch.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value


class OUNoise:
    def __init__(self, action_dimension, mu=0, theta=0.001, sigma=0.00925):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.action_dimension = action_dimension
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dimension)
        self.state = x + dx
        return self.state
    


if __name__ == '__main__':
    # Example usage
    ou_noise = OUNoise()
    current_value = 0.5
    for _ in range(10):
        noise = ou_noise.sample()
        current_value += noise  # Modifying some input value by adding noise
        print(current_value)
    