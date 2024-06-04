import numpy as np
import torch
from modules.networks_module import Actor, Critic, OUNoise
import torch.optim as optim


class Config:
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DISCOUNT = 0.99
    LEARNING_RATE = 0.001
    epsilon = 0.0  # Initial exploration probability
    epsilon_min = 0.01  # Minimum exploration probability
    epsilon_decay = 0.995  # Decay rate for exploration probability
    tau = 0.001
    num_episodes = 20000
    runtime = 1
    action_duration = 50 # 50 is equivalent to 1 cycle of pwm so less than that is not possible
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
    "127.0.0.108",
    "127.0.0.109",
    ]

    ips_str = ",".join(ips) # Join IP addresses into a single string

    buffer_capacity = len(ips) #1000000
    non_zero_ratio=0
    batch_coefficient = 10
    training_batch_size = batch_coefficient*len(ips)

    state_dim = 4
    action_dim = 1
    ou_noise = OUNoise(action_dimension=action_dim)
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
