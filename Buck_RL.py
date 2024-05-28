import os
import random
import socket
import struct
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed, wait

import matlab.engine
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from modules.plotting_module import plot_data
from modules.tcp_module import receive_data, send_data, websocket, TCP_PORT

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define the range of IP addresses
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

ips_str = ",".join(ips)  # Join IP addresses into a single string



# Initialize and hyperparameters
DISCOUNT = 0.99
LEARNING_RATE = 0.001
epsilon = 1.0  # Initial exploration probability
epsilon_min = 0.01  # Minimum exploration probability
epsilon_decay = 0.995  # Decay rate for exploration probability
tau = 0.001
num_episodes = 20000
runtime = 1
Vinit = 0
Iinit = 0
duty_step = np.linspace(0, 1, 201)

Vref = 5

# Define Actor and Critic networks


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
        x = torch.tanh(self.fc3(x)) * self.max_action
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


# Initialize networks
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

# Experience Replay


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


replay_buffer = ReplayBuffer()

# Define reward functions


def reward_stability(x):
    V = x[0]
    Vref = 5.0
    deviation = V - Vref
    penalty = deviation**2
    return -penalty


def reward_efficiency(u, prev_u):
    control_effort = (u - prev_u) ** 2
    return -0.01 * control_effort


def reward_convergence(current_deviation, prev_deviation):
    improvement = prev_deviation - current_deviation
    return improvement


def reward_time(t, max_time):
    return -t / max_time


def composite_reward(x, u, prev_u, prev_deviation, t, max_time):
    current_deviation = abs(x[0] - 5.0)
    stability = reward_stability(x)
    efficiency = reward_efficiency(u, prev_u)
    convergence = reward_convergence(current_deviation, prev_deviation)
    time_penalty = reward_time(t, max_time)

    weight_stability = 1.0
    weight_efficiency = 0.001
    weight_convergence = 0.5
    weight_time = 1.5  # Adjust this weight according to your preference

    total_reward = (
        weight_stability * stability
        + weight_efficiency * efficiency
        + weight_convergence * convergence
        + weight_time * time_penalty
    )

    return total_reward, current_deviation


# Define done function
class DoneChecker:
    def __init__(self):
        self.t0 = None
        self.desirable_band = [4.8, 5.2]

    def isdone(self, x, t):
        V = x[0]
        if V >= self.desirable_band[0] and V <= self.desirable_band[1]:
            if self.t0 is None:
                self.t0 = t
            elif t - self.t0 >= 0.5:
                return True
        else:
            self.t0 = None
        return False


done_checker = DoneChecker()

# Soft update for target networks


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


# Training DDPG model


def train_model(
    replay_buffer, batch_size=64, non_zero_ratio=0.8, csv_file="actions_log.csv"
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




# Function to run a single simulation episode


def run_simulations(model, ips_str):
    print("Starting MATLAB engine and running simulations...")
    eng = matlab.engine.start_matlab()
    eng.cd("/home/pvm8318/Documents/Python/Modular")
    eng.addpath("/home/pvm8318/Documents/Python/Modular")
    print(f"Running run_simulations with model: {model} and ips: {ips_str}")
    eng.run_simulations(model, ips_str, nargout=0)
    eng.quit()
    print("MATLAB simulations completed.")

    # Load and print simulation errors
    errors = scipy.io.loadmat("simulation_errors.mat")["errors"]
    # for i, error in enumerate(errors):
    #     print(f"Simulation {i + 1} error: {error}")




# Function to run a simulation episode


def run_simulation_episode(
    conn, ip, replay_buffer, episode, pause_event, train_event, lock
):
    print(f"Using established connection to {ip}:{TCP_PORT}")

    # Reset the environment and get the initial state
    state = np.array([Vinit, Iinit])
    total_reward = 0
    time = 0
    action = select_action(state)
    prev_deviation = 0
    prev_u = 0
    prev_u_step = 0
    prev_dev_step = 0
    Vo = []
    rewardval = []
    duty_cycle = []
    t = []
    iteration = 0

    action_duration = 200  # Hold action for 1000 steps

    while time < runtime:
        for _ in range(action_duration):
            send_data(conn, action)
            V, IL, Time = receive_data(conn)
            next_state = np.array([V, IL])

            # if iteration % 10 == 0:
            #     t.append(Time)
            #     Vo.append(V)
            #     rewardval.append(action)
            t.append(Time)
            Vo.append(V)
            duty_cycle.append(action)
            reward, current_deviation = composite_reward(
            next_state, action, prev_u, prev_deviation, time, runtime)
            rewardval.append(reward)
            prev_deviation = current_deviation
            prev_u = action
            time = Time
            iteration += 1

        reward, current_deviation = composite_reward(
            next_state, action, prev_u_step, prev_dev_step, time, runtime
        )
        prev_dev_step = current_deviation
        prev_u_step = action
        total_reward += reward

        with lock:
            replay_buffer.push(state, action, reward, next_state, False)

        state = next_state

        # if iteration % 200 == 0:
        #     pause_event.clear()  # Signal to pause
        #     train_event.wait()  # Wait for the training to be done
        #     pause_event.set()  # Signal to resume
        pause_event.clear()  # Signal to pause
        train_event.wait()  # Wait for the training to be done
        pause_event.set()  # Signal to resume
        # Select a new action after holding the previous one for 1000 steps
        action = select_action(state)

    conn.close()
    print(f"Completed episode {episode} for IP {ip}")

    return total_reward, t, Vo, duty_cycle, rewardval, episode


def train_network(
    replay_buffer, pause_event, train_event, lock, batch_size=512, terminate_event=None
):
    while not (terminate_event and terminate_event.is_set()):
        pause_event.wait()  # Wait for pause signal
        with lock:
            train_model(replay_buffer, batch_size=batch_size)
        train_event.set()  # Signal that training is done
        pause_event.wait()  # Wait for resume signal


# Main execution
if __name__ == "__main__":
    model = "Buck_Converter"
    replay_buffer = ReplayBuffer()
    episode_per_ip = num_episodes // len(ips)

    pause_event = threading.Event()
    train_event = threading.Event()
    lock = threading.Lock()
    terminate_event = threading.Event()

    pause_event.set()
    train_event.set()

    for batch in range(episode_per_ip):
        print(f"Starting batch {batch}")

        matlab_thread = threading.Thread(target=run_simulations, args=(model, ips_str))
        matlab_thread.start()

        print("Starting websocket connections...")

        connections = []
        with ThreadPoolExecutor(max_workers=len(ips)) as executor:
            futures = [executor.submit(websocket, ip, TCP_PORT) for ip in ips]
            for future in as_completed(futures):
                try:
                    conn = future.result()
                    connections.append(conn)
                except Exception as e:
                    print(f"Error establishing connection: {e}")

        if len(connections) != len(ips):
            print("Not all connections were established. Retrying the batch.")
            continue

        max_total_reward = float("-inf")
        best_t = []
        best_Vo = []
        best_rewardval = []
        episode = 0

        training_thread = threading.Thread(
            target=train_network,
            args=(replay_buffer, pause_event, train_event, lock, 64, terminate_event),
        )
        training_thread.start()

        with ThreadPoolExecutor(max_workers=len(ips)) as executor:
            futures = [
                executor.submit(
                    run_simulation_episode,
                    conn,
                    ip,
                    replay_buffer,
                    batch * len(ips) + i,
                    pause_event,
                    train_event,
                    lock,
                )
                for i, (conn, ip) in enumerate(zip(connections, ips))
            ]
            for future in as_completed(futures):
                total_reward, t, Vo, duty_cycle, rewardval, Episode = future.result()
                if total_reward > max_total_reward:
                    max_total_reward = total_reward
                    best_t = t
                    best_Vo = Vo
                    best_duty_cycle = duty_cycle
                    best_rewardval = rewardval
                    episode = Episode

        matlab_thread.join()
        terminate_event.set()
        training_thread.join()

        # Plot the best episode in the current batch
        plot_data(best_t, best_Vo, best_duty_cycle, best_rewardval, episode, max_total_reward)

        print(f"Completed training for batch {batch}")
