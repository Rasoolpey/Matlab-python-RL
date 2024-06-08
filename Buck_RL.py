import torch
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Barrier, Lock

from modules.config_module import Config
from modules.decision_evaluation_module import DoneChecker
from modules.plotting_module import plot_data
from modules.tcp_module import TCP_PORT, websocket
from modules.simulation_module import run_simulations, run_simulation_episode
from modules.rl_training_utils import ReplayBuffer, train_network, actor, critic


config = Config()
num_episodes = config.num_episodes
Vref = config.Vref
ips = config.ips
ips_str = config.ips_str
device = config.device
save_path = config.save_path
training_batch_size = config.training_batch_size
non_zero_ratio = config.non_zero_ratio
lock = Lock()

# Experience Replay
replay_buffer = ReplayBuffer()

barrier = Barrier(parties=len(ips), action=lambda: train_network(replay_buffer, lock, batch_size=training_batch_size, non_zero_ratio=non_zero_ratio, device=device))
# Main execution

if __name__ == "__main__":
    model = "Buck_Converter"
    episode_per_ip = num_episodes // len(ips)
    done_checker = DoneChecker(Vref)
    
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

        # training_thread = threading.Thread(
        #     target=train_network,
        #     args=(replay_buffer, lock,training_batch_size, terminate_event, device),
        # )
        # training_thread.start()

        with ThreadPoolExecutor(max_workers=len(ips)) as executor:
            futures = [
                executor.submit(
                    run_simulation_episode,
                    conn,
                    ip,
                    replay_buffer,
                    batch * len(ips) + i,
                    lock,
                    barrier,
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
                if total_reward > -1500:
                    torch.save(actor.state_dict(), save_path + f"{total_reward}, episode : {Episode},  actor.pth")
                    torch.save(critic.state_dict(), save_path + f"{total_reward}, episode : {Episode}, critic.pth")

        # with ThreadPoolExecutor(max_workers=len(ips)) as executor:
        #     # Submit tasks to the executor for each IP
        #     futures = {}
        #     for i, (conn, ip) in enumerate(zip(connections, ips)):
        #         future = executor.submit(
        #             run_simulation_episode,
        #             conn,
        #             ip,
        #             replay_buffer,
        #             batch * len(ips) + i,
        #             lock,
        #             barrier,
        #         )
        #         futures[future] = ip  # Store future with associated IP for tracking
        #         print(f"Started training for IP: {ip} in batch {batch}")  # Indicate start for each IP

        #     # Collect results as tasks complete
        #     for future in as_completed(futures):
        #         ip = futures[future]  # Retrieve the IP associated with this future
        #         try:
        #             total_reward, t, Vo, duty_cycle, rewardval, Episode = future.result()
        #             print(f"Completed training for IP: {ip} in batch {batch} - Episode: {Episode} - Reward: {total_reward}")  # Indicate completion for each IP

        #             # Check if the current run has the highest total reward
        #             if total_reward > max_total_reward:
        #                 max_total_reward = total_reward
        #                 best_t = t
        #                 best_Vo = Vo
        #                 best_duty_cycle = duty_cycle
        #                 best_rewardval = rewardval
        #                 episode = Episode
        #                 print(f"New best training session found: IP {ip} in batch {batch} with total reward: {max_total_reward}")
                        
        #         except Exception as e:
        #             print(f"Error during training for IP: {ip} in batch {batch}: {e}")  # Indicate error if occurred

        matlab_thread.join()
        # terminate_event.set()
        # training_thread.join()

        # Plot the best episode in the current batch
        plot_data(
            best_t, best_Vo, best_duty_cycle, best_rewardval, episode, max_total_reward
        )

        print(f"Completed training for batch {batch}")
