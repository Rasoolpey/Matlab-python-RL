# Reward calculation functions
def calculate_stability_reward(Vref,x):
    V = x[0]
    vref = Vref
    deviation = V - vref
    penalty = deviation**2
    return -penalty

def calculate_efficiency_reward(u, prev_u):
    control_effort = (u - prev_u) ** 2
    return -0.01 * control_effort

def calculate_convergence_reward(current_deviation, prev_deviation):
    improvement = prev_deviation - current_deviation
    return improvement

def calculate_time_reward(t, max_time):
    return -t / max_time

def composite_reward(Vref, x, u, prev_u, prev_deviation, t, max_time):
    current_deviation = abs(x[0] - Vref)
    stability = calculate_stability_reward(Vref,x)
    efficiency = calculate_efficiency_reward(u, prev_u)
    convergence = calculate_convergence_reward(current_deviation, prev_deviation)
    time_penalty = calculate_time_reward(t, max_time)

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
 # plt.savefig(f"plots/episode_number_{episode}.png", dpi=600)
    return total_reward, current_deviation

# Done checking class
class DoneChecker:
    def __init__(self, Vref=7.5):
        self.t0 = None
        self.desirable_band = [0.97*Vref, 1.03*Vref]

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

# Decision evaluation module
if __name__ == "__main__":
    # Example usage of the evaluation functions
    x = [5.1]  # Example state
    u = 0.5  # Example control input
    prev_u = 0.4  # Example previous control input
    prev_deviation = 0.2  # Example previous deviation
    t = 10  # Example time
    max_time = 100  # Example max time
    Vref = 5.0  # Example reference voltage

    # Calculate composite reward
    reward, current_deviation = composite_reward(Vref, x, u, prev_u, prev_deviation, t, max_time)
    print("Composite Reward:", reward)

    # Create an instance of DoneChecker
    done_checker = DoneChecker()

    # Check if done
    done = done_checker.isdone(x, t)
    print("Is Done:", done)
