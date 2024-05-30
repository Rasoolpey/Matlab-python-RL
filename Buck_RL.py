
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from modules.config_module import Config
from modules.decision_evaluation_module import DoneChecker
from modules.plotting_module import plot_data
from modules.tcp_module import TCP_PORT, websocket
from modules.simulation_module import run_simulations, run_simulation_episode
from modules.rl_training_utils import ReplayBuffer, train_network


config = Config()
num_episodes = config.num_episodes
Vref = config.Vref
ips = config.ips
ips_str = config.ips_str
device = config.device


# Experience Replay
replay_buffer = ReplayBuffer()


# Main execution
if __name__ == "__main__":
    model = "Buck_Converter"
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
        plot_data(
            best_t, best_Vo, best_duty_cycle, best_rewardval, episode, max_total_reward
        )

        print(f"Completed training for batch {batch}")
