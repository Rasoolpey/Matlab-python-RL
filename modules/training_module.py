import torch
import numpy as np

from training import actor, select_action, train_model
from replay_buffer import replay_buffer
from init_hyperparameters import runtime, Vinit, Iinit, DISCOUNT, epsilon, epsilon_min, epsilon_decay



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
    replay_buffer, pause_event, train_event, lock, batch_size=512, terminate_event=None
):
    while not (terminate_event and terminate_event.is_set()):
        pause_event.wait()  # Wait for pause signal
        with lock:
            train_model(replay_buffer, batch_size=batch_size)
        train_event.set()  # Signal that training is done
        pause_event.wait()  # Wait for resume signal


import torch
import torch.optim as optim
import torch.nn as nn
from init_hyperparameters import DISCOUNT, LEARNING_RATE, tau
from networks import Actor, Critic
from replay_buffer import ReplayBuffer

# Initialize networks
state_dim = 2
action_dim = 1
max_action = 1.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Soft update for target networks
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

# Training DDPG model
def train_model(
    replay_buffer, batch_size=64, non_zero_ratio=0.8
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

    # Get target Q-values
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
