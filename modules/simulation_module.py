import matlab.engine
import numpy as np
import scipy.io
from modules.tcp_module import receive_data, send_data, TCP_PORT
from modules.decision_evaluation_module import composite_reward
from modules.config_module import Config
from modules.rl_training_utils import select_action



config = Config()

num_episodes = config.num_episodes
runtime = config.runtime
Vinit = config.Vinit
Iinit = config.Iinit
Vref = config.Vref
device = config.device
action_duration = config.action_duration
batch_coefficient = config.batch_coefficient
ou_noise = config.ou_noise


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


def run_simulation_episode(conn, ip, replay_buffer, episode, lock, barrier, batch_coefficient=batch_coefficient):
    print(f"Using established connection to {ip}:{TCP_PORT}")

    # Reset the environment and get the initial state
    state = np.array([Vinit, Iinit, 0, 0])
    next_state = np.array([Vinit, Iinit])
    prev_state = np.array([Vinit, Iinit, 0, 0])
    dev_state = np.array([0, 0])
    total_reward = 0
    time = 0
    action = select_action(state, ou_noise)
    prev_deviation = 0
    prev_u = 0
    prev_u_step = 0
    prev_dev_step = 0
    Vo = []
    rewardval = []
    duty_cycle = []
    t = []
    iteration = 0


    while time < runtime:
        for _ in range(action_duration):
            send_data(conn, action)
            V, IL, Time = receive_data(conn)
            next_state = np.array([V, IL])
            dev_state = next_state - prev_state[:2]
            # if iteration % 10 == 0:
            #     t.append(Time)
            #     Vo.append(V)
            #     rewardval.append(action)
            # Update the state to include rate of change
            state = np.concatenate([next_state, dev_state])

            t.append(Time)
            Vo.append(V)
            duty_cycle.append(action)
            reward, current_deviation = composite_reward(
            Vref,next_state, action, prev_u, prev_deviation, time, runtime)
            rewardval.append(reward)
            prev_deviation = current_deviation
            prev_u = action
            time = Time
            iteration += 1

        reward, current_deviation = composite_reward(
            Vref,next_state, action, prev_u_step, prev_dev_step, time, runtime
        )
        prev_dev_step = current_deviation
        prev_u_step = action
        total_reward += reward
        with lock:
            replay_buffer.push(prev_state, action, reward, state, False)
            
        prev_state = state
        # if iteration % 200 == 0:
        #     pause_event.clear()  # Signal to pause
        #     train_event.wait()  # Wait for the training to be done
        #     pause_event.set()  # Signal to resume
        # pause_event.clear()  # Signal to pause
        # train_event.wait()  # Wait for the training to be done
        # pause_event.set()  # Signal to resume
        # print(f"Completed iteration {iteration} for IP {ip}")
        if iteration % batch_coefficient == 0:
            barrier.wait()
        # print(f"Continuing to next iteration for IP {ip}")
        # Select a new action after holding the previous one for 1000 steps
        # action = select_action(state)
        action = select_action(state, ou_noise)
        # print(f"Selected new action: {action}")

    conn.close()
    print(f"Completed episode {episode} for IP {ip}")

    return total_reward, t, Vo, duty_cycle, rewardval, episode