import numpy as np
import torch
from modules.networks_module import Actor, Critic
import torch.optim as optim


class Config:
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DISCOUNT = 0.99
    LEARNING_RATE = 0.001
    epsilon = 1.0  # Initial exploration probability
    epsilon_min = 0.01  # Minimum exploration probability
    epsilon_decay = 0.995  # Decay rate for exploration probability
    tau = 0.001
    num_episodes = 20000
    runtime = 5
    action_duration = 4000 # Hold action for 2000 steps
    Vinit = 0
    Iinit = 0
    duty_step = np.linspace(0, 1, 201)
    Vref = 7.5

    ips = [
    "127.0.0.100",
    "127.0.0.101",
    "127.0.0.102",
    "127.0.0.103",
    "127.0.0.104",
    "127.0.0.105",
    "127.0.0.106",
    "127.0.0.107",
    ]

    ips_str = ",".join(ips) # Join IP addresses into a single string

    state_dim = 2
    action_dim = 1
    max_action = 1.0
    actor = Actor(state_dim, action_dim, max_action).to(device)
    actor_target = Actor(state_dim, action_dim, max_action).to(device)
    critic = Critic(state_dim, action_dim).to(device)
    critic_target = Critic(state_dim, action_dim).to(device)

    # Initialize target network weights
    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())

    # Optimizers
    actor_optimizer = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
    critic_optimizer = optim.Adam(critic.parameters(), lr=LEARNING_RATE)
