
import random
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from modules.networks_module import Actor, Critic
from modules.config_module import Config

config = Config()
device = config.device
DISCOUNT = config.DISCOUNT
LEARNING_RATE = config.LEARNING_RATE
epsilon = config.epsilon
epsilon_min = config.epsilon_min
epsilon_decay = config.epsilon_decay
tau = config.tau
num_episodes = config.num_episodes
runtime = config.runtime
Vinit = config.Vinit
Iinit = config.Iinit
duty_step = config.duty_step
Vref = config.Vref
state_dim = config.state_dim
action_dim = config.action_dim
max_action = config.max_action
non_zero_ratio=config.non_zero_ratio

print(f'using device:', device)

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


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size, non_zero_ratio=0.8):
        non_zero_samples = [
            experience for experience in self.buffer if experience[1] != 0
        ]
        zero_samples = [experience for experience in self.buffer if experience[1] == 0]

        non_zero_sample_size = int(batch_size * non_zero_ratio)
        zero_sample_size = batch_size - non_zero_sample_size

        non_zero_sample_size = min(len(non_zero_samples), non_zero_sample_size)
        zero_sample_size = min(len(zero_samples), zero_sample_size)

        samples = random.sample(non_zero_samples, non_zero_sample_size) + random.sample(
            zero_samples, zero_sample_size
        )
        random.shuffle(samples)

        states, actions, rewards, next_states, dones = map(np.array, zip(*samples))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        self.buffer = []




# Soft update for target networks


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


# Training DDPG model


def train_model(
    replay_buffer, batch_size=64, non_zero_ratio=0.8, csv_file="actions_log.csv", device=device
):
    if len(replay_buffer) < batch_size:
        return
    states, actions, rewards, next_states, dones = replay_buffer.sample(
        batch_size, non_zero_ratio
    )

    states = torch.FloatTensor(states).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    actions = torch.FloatTensor(actions).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
    dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

    # Get target Q-values):
    next_actions = actor_target(next_states)
    next_q_values = critic_target(next_states, next_actions)
    target_q_values = rewards + DISCOUNT * next_q_values * (1 - dones)

    # Critic loss
    q_values = critic(states, actions)
    critic_loss = nn.MSELoss()(q_values, target_q_values.detach())
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # Actor loss
    actor_loss = -critic(states, actor(states)).mean()
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    # Soft update target networks
    soft_update(actor_target, actor, tau)
    soft_update(critic_target, critic, tau)
   



# Select action with exploration noise
def select_action(state, noise_scale=0.1):
    global epsilon
    state = torch.FloatTensor(state).unsqueeze(0).to(device)

    if np.random.rand() < epsilon:
        # Exploration: choose a random action
        action = np.random.uniform(0, 1, action_dim)
    else:
        # Exploitation: choose the action suggested by the actor network
        action = actor(state).cpu().detach().numpy()[0]
        action = np.clip(action + noise_scale * np.random.randn(action_dim), 0, 1)

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return action.item()


def train_network(
    replay_buffer, lock, batch_size=512, terminate_event=None, device=device
):
    if terminate_event is not None and terminate_event.is_set():
        print("Training skipped because terminate event is set.")
    else:
        with lock:
            train_model(replay_buffer, batch_size=batch_size, non_zero_ratio=non_zero_ratio, device=device)
        replay_buffer.clear()  # Clear the buffer after training.
        # print("Training done.")
