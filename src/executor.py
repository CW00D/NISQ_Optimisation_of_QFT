import time
import os
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import simple_optimiser
from qiskit.quantum_info import Statevector
import random
import numpy as np

# ================================
# Execution Parameters (Global)
# ================================
population = 20
qubits = 3
initial_circuit_depth = 10

# Mutation Parameters
elitism_number = population // 3      # 3
parameter_mutation_rate = 0.1         # 0.1
gate_mutation_rate = 0.3              # 0.3
layer_mutation_rate = 0.2             # 0.2
max_parameter_mutation = 0.2          # 0.2
layer_deletion_rate = 0.03            # 0.03

# Random Baseline Parameters
BASELINE_ITERATIONS = 20000
N_RANDOM_RUNS = 10
RANDOM_BASELINE_DIR = "Experiment Results/Random_Baseline"
RANDOM_BASELINE_FILE = os.path.join(RANDOM_BASELINE_DIR, "random_baseline.csv")

# ================================
# Random Baseline Generation & Management
# ================================
def compute_random_baseline(algorithm_type, baseline_iterations=BASELINE_ITERATIONS, n_random_runs=N_RANDOM_RUNS):
    """
    Computes the random search baseline over baseline_iterations averaged over n_random_runs.
    Returns two lists: avg_random_max and avg_random_avg.
    """
    # Precompute the initial states and target states once.
    initial_states = [Statevector.from_label(f"{i:0{qubits}b}") for i in range(2**qubits)]
    target_states = algorithm_type.get_qft_target_states(qubits)
    
    all_random_max = []
    all_random_avg = []
    for run in range(n_random_runs):
        print(f"Random baseline run {run}")
        current_random_max = 0
        run_random_max = []
        run_random_avg = []
        for i in range(baseline_iterations):
            random_max_fitness, random_avg_fitness = algorithm_type.evaluate_random_circuits(
                population, 1, qubits, initial_circuit_depth, initial_states, target_states
            )
            current_random_max = max(current_random_max, random_max_fitness)
            run_random_max.append(current_random_max)
            run_random_avg.append(random_avg_fitness)
        all_random_max.append(run_random_max)
        all_random_avg.append(run_random_avg)
    avg_random_max = [sum(run[i] for run in all_random_max) / n_random_runs for i in range(baseline_iterations)]
    avg_random_avg = [sum(run[i] for run in all_random_avg) / n_random_runs for i in range(baseline_iterations)]
    return avg_random_max, avg_random_avg

def save_random_baseline(baseline_max, baseline_avg):
    os.makedirs(RANDOM_BASELINE_DIR, exist_ok=True)
    with open(RANDOM_BASELINE_FILE, "w") as f:
        f.write("Iteration,Random_Max,Random_Avg\n")
        for i in range(len(baseline_max)):
            f.write(f"{i},{baseline_max[i]:.6f},{baseline_avg[i]:.6f}\n")

def load_random_baseline(n_iterations):
    baseline_max = []
    baseline_avg = []
    if not os.path.exists(RANDOM_BASELINE_FILE):
        return None, None
    with open(RANDOM_BASELINE_FILE, "r") as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            iteration, rmax, ravg = parts
            baseline_max.append(float(rmax))
            baseline_avg.append(float(ravg))
    return baseline_max[:n_iterations], baseline_avg[:n_iterations]

def ensure_random_baseline(algorithm_type, baseline_iterations=BASELINE_ITERATIONS, n_random_runs=N_RANDOM_RUNS):
    """
    Checks for a precomputed random baseline file. If not found, computes and saves it.
    Returns the full baseline arrays.
    """
    if os.path.exists(RANDOM_BASELINE_FILE):
        print("Loading existing random baseline...")
        baseline_max, baseline_avg = load_random_baseline(BASELINE_ITERATIONS)
        if baseline_max is not None:
            return baseline_max, baseline_avg
    print("Computing random baseline...")
    baseline_max, baseline_avg = compute_random_baseline(algorithm_type, baseline_iterations, n_random_runs)
    save_random_baseline(baseline_max, baseline_avg)
    return baseline_max, baseline_avg

# ================================
# Plotting Logic
# ================================
def plot_iteration_results(timestamp, ea_max, ea_avg, random_max, random_avg, x_values):
    plt.clf()
    plt.plot(x_values, ea_max, label="EA Max Fitness", color="blue")
    plt.plot(x_values, ea_avg, label="EA Avg Fitness", color="blue", linestyle="--")
    plt.plot(x_values, random_max, label="Random Max Overall Fitness", color="orange")
    plt.plot(x_values, random_avg, label="Random Avg Fitness", color="orange", linestyle="--")
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.ylim(0, 1)
    plt.title("Fitness Over Iterations (Averaged over EA runs)")
    plt.grid(True)
    out_dir = f"Experiment Results/Charts/{simple_optimiser.__name__.capitalize()}"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/{timestamp}.png")
    plt.close()

# ================================
# EA Execution
# ================================
def execute_optimisation(timestamp, algorithm_type, iterations, n_runs=10):
    # Create folder for log files.
    results_folder = f"Experiment Results/Logs/{algorithm_type.__name__.capitalize()}"
    os.makedirs(results_folder, exist_ok=True)
    log_filepath = os.path.join(results_folder, f"{timestamp}.log")
    
    # Precompute initial and target states.
    initial_states = [Statevector.from_label(f"{i:0{qubits}b}") for i in range(2**qubits)]
    target_states = algorithm_type.get_qft_target_states(qubits)
    
    # Ensure we have a precomputed random baseline.
    baseline_max, baseline_avg = ensure_random_baseline(algorithm_type)
    
    all_ea_max = []
    all_ea_avg = []
    for run in range(n_runs):
        seed_value = int(time.time()) + run
        random.seed(seed_value)
        np.random.seed(seed_value)
        
        chromosomes = algorithm_type.initialize_chromosomes(population, qubits, initial_circuit_depth)
        run_ea_max = []
        run_ea_avg = []
        for i in range(iterations):
            if i % 100 == 0:
                print(f"EA Run {run}, iteration {i}")
            circuits = algorithm_type.get_circuits(chromosomes)
            fitnesses = algorithm_type.get_circuit_fitnesses(target_states, circuits, initial_states)
            max_fitness = max(fitnesses)
            avg_fitness = sum(fitnesses) / len(fitnesses)
            run_ea_max.append(max_fitness)
            run_ea_avg.append(avg_fitness)
            
            if max_fitness >= 1:  # Stop early if perfect circuit is found.
                remaining = iterations - i - 1
                run_ea_max.extend([max_fitness] * remaining)
                run_ea_avg.extend([avg_fitness] * remaining)
                break
            
            chromosomes = algorithm_type.apply_genetic_operators(
                chromosomes, fitnesses, elitism_number,
                parameter_mutation_rate, gate_mutation_rate,
                layer_mutation_rate, max_parameter_mutation,
                layer_deletion_rate
            )
        all_ea_max.append(run_ea_max)
        all_ea_avg.append(run_ea_avg)
    
    # Average the EA results over the n_runs.
    avg_ea_max = [sum(run[i] for run in all_ea_max) / n_runs for i in range(iterations)]
    avg_ea_avg = [sum(run[i] for run in all_ea_avg) / n_runs for i in range(iterations)]
    x_values = list(range(iterations))
    
    # Write EA and random baseline (first n iterations) to log file.
    with open(log_filepath, "w") as log_file:
        log_file.write("Iteration,EA Max Fitness,EA Avg Fitness,Random Max Fitness,Random Avg Fitness\n")
        for i in range(iterations):
            log_file.write(f"{i},{avg_ea_max[i]:.6f},{avg_ea_avg[i]:.6f},{baseline_max[i]:.6f},{baseline_avg[i]:.6f}\n")
    
    plot_iteration_results(timestamp, avg_ea_max, avg_ea_avg, baseline_max[:iterations], baseline_avg[:iterations], x_values)
    print("Optimisation complete. Averaged results saved in:", log_filepath)

# ================================
# Main Execution
# ================================
if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    # For example, run EA for 20,000 iterations (should be <= BASELINE_ITERATIONS)
    execute_optimisation(timestamp, simple_optimiser, iterations=100, n_runs=1)
