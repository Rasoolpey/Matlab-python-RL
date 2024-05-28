import threading
import random
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch
import torch.nn as nn
from modules.plotting_module import plot_data
from modules.tcp_module import websocket, TCP_PORT
from modules.decision_evaluation_module import DoneChecker
from modules.config_module import Config
from modules.simulation_module import run_simulations, run_simulation_episode


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
print("Using device:", device)

# Initialize actor and critic networks
actor = config.actor
actor_target = config.actor_target
critic = config.critic
critic_target = config.critic_target

# Initialize target network weights
actor_target.load_state_dict(actor.state_dict())
critic_target.load_state_dict(critic.state_dict())

# Optimizers
actor_optimizer = config.actor_optimizer
critic_optimizer = config.critic_optimizer


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
    done_checker = DoneChecker(Vref)
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
