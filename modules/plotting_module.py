import os
import matplotlib.pyplot as plt

# def plot_data(time, Vo, duty_cycle, cost, episode, total_reward):
#     # Ensure the plots directory exists
#     if not os.path.exists("plots"):
#         os.makedirs("plots")

#     # Create a new figure with two subplots
#     fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
#     fig.suptitle(f"Episode {episode} - Total Reward: {total_reward:.2f}")

#     # Top subplot: Output Voltage and Duty Cycle
#     ax1.set_title("Output Voltage and Duty Cycle")
#     ax1.plot(time, Vo, color="orangered", label="Output Voltage")
#     ax1.set_ylabel("Output Voltage", color="orangered")
#     ax1.tick_params(axis="y", colors="orangered")

#     ax1_twin = ax1.twinx()
#     ax1_twin.plot(time, duty_cycle, color="steelblue", label="Duty Cycle")
#     ax1_twin.set_ylabel("Duty Cycle", color="steelblue")
#     ax1_twin.tick_params(axis="y", colors="steelblue")

#     # Customize spines colors
#     ax1.spines["top"].set_color("gray")
#     ax1.spines["bottom"].set_color("black")
#     ax1.spines["left"].set_color("orangered")
#     ax1_twin.spines["right"].set_color("steelblue")

#     # Bottom subplot: Cost
#     ax2.set_title("Cost Over Time")
#     ax2.plot(time, cost, color="green", label="Cost")
#     ax2.set_xlabel("Time")
#     ax2.set_ylabel("Cost", color="green")
#     ax2.tick_params(axis="y", colors="green")

#     # Adjust layout and save the figure with high resolution
#     plt.tight_layout(rect=[0, 0, 1, 0.96])
#     # plt.savefig(f"plots/episode_number_{episode}.png", dpi=600)
#     plt.savefig(f"plots/episode_number_{episode}.svg", format='svg')
#     plt.close()


# if __name__ == "__main__":
#     # Example usage
#     import numpy as np

#     time = np.linspace(0, 10, 100)
#     Vo = np.random.rand(100)
#     duty_cycle = np.random.rand(100)
#     cost = np.random.rand(100)
#     episode = 1
#     total_reward = 100

#     plot_data(time, Vo, duty_cycle, cost, episode, total_reward)



def plot_data(time, Vo, duty_cycle, cost, episode, total_reward):
    # Ensure the plots directory exists
    if not os.path.exists("plots"):
        os.makedirs("plots")

    # Create a new figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle(f"Episode {episode} - Total Reward: {total_reward:.2f}")

    # First subplot: Output Voltage
    ax1.set_title("Output Voltage")
    ax1.plot(time, Vo, color="orangered", label="Output Voltage")
    ax1.set_ylabel("Output Voltage", color="orangered")
    ax1.tick_params(axis="y", colors="orangered")

    # Second subplot: Duty Cycle
    ax2.set_title("Duty Cycle")
    ax2.plot(time, duty_cycle, color="steelblue", label="Duty Cycle")
    ax2.set_ylabel("Duty Cycle", color="steelblue")
    ax2.tick_params(axis="y", colors="steelblue")

    # Third subplot: Cost
    ax3.set_title("Cost Over Time")
    ax3.plot(time, cost, color="green", label="Cost")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Cost", color="green")
    ax3.tick_params(axis="y", colors="green")

    # Adjust layout and save the figure with high resolution
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"plots/episode_number_{episode}.svg", format='svg')
    plt.close()

if __name__ == "__main__":
    # Example usage
    import numpy as np

    time = np.linspace(0, 10, 100)
    Vo = np.random.rand(100)
    duty_cycle = np.random.rand(100)
    cost = np.random.rand(100)
    episode = 1
    total_reward = 100

    plot_data(time, Vo, duty_cycle, cost, episode, total_reward)
