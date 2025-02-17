import time
import os
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import simple_optimiser
#from simple_optimiser_noisy import *
#from moea_optimiser import *
#from moea_optimiser_noisy import *

from qiskit.quantum_info import Statevector
import random
import numpy as np

# Simulation Parameters
population = 20
qubits = 3
initial_circuit_depth = 10

# Mutation Parameters
elitism_number = population // 3    # 3
parameter_mutation_rate = 0.1         # 0.1
gate_mutation_rate = 0.3              # 0.3
layer_mutation_rate = 0.2             # 0.2
max_parameter_mutation = 0.2          # 0.2
layer_deletion_rate = 0.03            # 0.03

# Parameters for random baseline computation
BASELINE_ITERATIONS = 20000
N_RANDOM_RUNS = 10
RANDOM_BASELINE_DIR = "Experiment Results/Random_Baseline"
RANDOM_BASELINE_FILE = os.path.join(RANDOM_BASELINE_DIR, "random_baseline.csv")

# Random Basaeline Generation and Management
def compute_random_baseline(algorithm_type, population, qubits, initial_circuit_depth, initial_states, target_states,
                            baseline_iterations=BASELINE_ITERATIONS, n_random_runs=N_RANDOM_RUNS):
    """
    Computes the random search baseline over baseline_iterations averaged over n_random_runs.
    Returns two lists: avg_random_max and avg_random_avg (each of length baseline_iterations).
    """
    all_random_max = []
    all_random_avg = []
    for run in range(n_random_runs):
        print(run)
        current_random_max = 0
        run_random_max = []
        run_random_avg = []
        for i in range(baseline_iterations):
            # Here we run one iteration of random search.
            random_max_fitness, random_avg_fitness = algorithm_type.evaluate_random_circuits(
                population, 1, qubits, initial_circuit_depth, initial_states, target_states
            )
            current_random_max = max(current_random_max, random_max_fitness)
            run_random_max.append(current_random_max)
            run_random_avg.append(random_avg_fitness)
        all_random_max.append(run_random_max)
        all_random_avg.append(run_random_avg)
    # Compute elementwise average over runs.
    avg_random_max = [sum(run[i] for run in all_random_max) / n_random_runs for i in range(baseline_iterations)]
    avg_random_avg = [sum(run[i] for run in all_random_avg) / n_random_runs for i in range(baseline_iterations)]
    return avg_random_max, avg_random_avg

def save_random_baseline(baseline_max, baseline_avg, baseline_iterations=BASELINE_ITERATIONS):
    os.makedirs(RANDOM_BASELINE_DIR, exist_ok=True)
    with open(RANDOM_BASELINE_FILE, "w") as f:
        f.write("Iteration,Random_Max,Random_Avg\n")
        for i in range(baseline_iterations):
            f.write(f"{i},{baseline_max[i]:.6f},{baseline_avg[i]:.6f}\n")

def load_random_baseline(n_iterations):
    """
    Loads the random baseline from file and returns the first n_iterations values.
    """
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
    # Return only up to n_iterations
    return baseline_max[:n_iterations], baseline_avg[:n_iterations]

def ensure_random_baseline(algorithm_type, population, qubits, initial_circuit_depth, initial_states, target_states, baseline_iterations=BASELINE_ITERATIONS, n_random_runs=N_RANDOM_RUNS):
    """
    Checks if the random baseline file exists. If not, computes and saves it.
    Returns the full baseline arrays.
    """
    if os.path.exists(RANDOM_BASELINE_FILE):
        print("Loading existing random baseline...")
        baseline_max, baseline_avg = load_random_baseline(baseline_iterations)
        if baseline_max is not None:
            return baseline_max, baseline_avg
    print("Computing random baseline...")
    baseline_max, baseline_avg = compute_random_baseline(algorithm_type, population, qubits, initial_circuit_depth,
                                                          initial_states, target_states,
                                                          baseline_iterations, n_random_runs)
    save_random_baseline(baseline_max, baseline_avg, baseline_iterations)
    return baseline_max, baseline_avg

# Optimisation Management Logic
def execute_optimisation(timestamp, algorithm_type, mode, population, qubits, initial_circuit_depth, elitism_number, iterations=-1, runtime_minutes=-1, n_runs = 10):
    # Create folder for log files.
    results_folder = f"Experiment Results/Logs/{algorithm_type.__name__.capitalize()}/{mode.capitalize()}_Execution"
    os.makedirs(results_folder, exist_ok=True)
    log_filename = f"{timestamp}.log"
    log_filepath = os.path.join(results_folder, log_filename)
    
    initial_states = [Statevector.from_label(f"{i:0{qubits}b}") for i in range(2**qubits)]
    target_states = algorithm_type.get_qft_target_states(qubits)
    
    if mode == "iteration":
        baseline_max, baseline_avg = ensure_random_baseline(algorithm_type, population, qubits,
                                                             initial_circuit_depth, initial_states, target_states)
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
                    print(f"Run {run}, iteration {i}")
                circuits = algorithm_type.get_circuits(chromosomes)
                fitnesses = algorithm_type.get_circuit_fitnesses(target_states, circuits, initial_states)
                max_fitness = max(fitnesses)
                avg_fitness = sum(fitnesses) / len(fitnesses)
                run_ea_max.append(max_fitness)
                run_ea_avg.append(avg_fitness)
                
                # Stop early if a perfect circuit is found; pad the remainder.
                if max_fitness >= 1:
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
        
        # Average the EA results across runs.
        avg_ea_max = [sum(run[i] for run in all_ea_max) / n_runs for i in range(iterations)]
        avg_ea_avg = [sum(run[i] for run in all_ea_avg) / n_runs for i in range(iterations)]
        x_values = list(range(iterations))
        
        # Write EA and random baseline (first n iterations) to the log file.
        with open(log_filepath, "w") as log_file:
            log_file.write("Iteration,EA Max Fitness,EA Avg Fitness,Random Max Fitness,Random Avg Fitness\n")
            for i in range(iterations):
                log_file.write(f"{i},{avg_ea_max[i]:.6f},{avg_ea_avg[i]:.6f},{baseline_max[i]:.6f},{baseline_avg[i]:.6f}\n")
        
        plot_iteration_results(timestamp, algorithm_type, avg_ea_max, avg_ea_avg, baseline_max[:iterations], baseline_avg[:iterations], x_values)
    
    elif mode == "timed":
        # Timed mode can be updated in a similar fashion if desired.
        all_ea_max = []
        all_ea_avg = []
        all_ea_x = []
        # For timed mode we assume we still compute random baseline on the fly
        # (or you could similarly load from file).
        all_random_max = []
        all_random_avg = []
        all_random_x = []
        n_runs = 10
        for run in range(n_runs):
            seed_value = int(time.time()) + run
            random.seed(seed_value)
            np.random.seed(seed_value)
            
            chromosomes = algorithm_type.initialize_chromosomes(population, qubits, initial_circuit_depth)
            target_states = algorithm_type.get_qft_target_states(qubits)
            
            # Random evaluation for first half of runtime.
            start_time = time.time()
            half_runtime = (runtime_minutes * 60) / 2
            log_interval = 10  # seconds
            run_random_max = []
            run_random_avg = []
            run_random_x = []
            current_random_max = 0
            values_logged_random = 0
            elapsed_time = 0
            while elapsed_time < half_runtime:
                elapsed_time = time.time() - start_time
                random_max_fitness, random_avg_fitness = algorithm_type.evaluate_random_circuits(
                    population, 1, qubits, initial_circuit_depth, initial_states, target_states
                )
                current_random_max = max(current_random_max, random_max_fitness)
                if elapsed_time >= (log_interval * values_logged_random):
                    run_random_x.append(elapsed_time)
                    run_random_max.append(current_random_max)
                    run_random_avg.append(random_avg_fitness)
                    values_logged_random += 1
            
            # EA evaluation for second half of runtime.
            ea_start_time = time.time()
            run_ea_max = []
            run_ea_avg = []
            run_ea_x = []
            values_logged_ea = 0
            elapsed_time_ea = 0
            while elapsed_time_ea < half_runtime:
                elapsed_time_ea = time.time() - ea_start_time
                circuits = algorithm_type.get_circuits(chromosomes)
                fitnesses = algorithm_type.get_circuit_fitnesses(target_states, circuits, initial_states)
                max_fitness = max(fitnesses)
                avg_fitness = sum(fitnesses) / len(fitnesses)
                if elapsed_time_ea >= (log_interval * values_logged_ea):
                    run_ea_x.append(elapsed_time_ea)
                    run_ea_max.append(max_fitness)
                    run_ea_avg.append(avg_fitness)
                    values_logged_ea += 1
                chromosomes = algorithm_type.apply_genetic_operators(
                    chromosomes, fitnesses, elitism_number,
                    parameter_mutation_rate, gate_mutation_rate,
                    layer_mutation_rate, max_parameter_mutation,
                    layer_deletion_rate
                )
            
            all_random_max.append(run_random_max)
            all_random_avg.append(run_random_avg)
            all_random_x.append(run_random_x)
            all_ea_max.append(run_ea_max)
            all_ea_avg.append(run_ea_avg)
            all_ea_x.append(run_ea_x)
        
        num_random_points = len(all_random_x[0])
        num_ea_points = len(all_ea_x[0])
        avg_random_x = all_random_x[0]
        avg_ea_x = all_ea_x[0]
        
        avg_ea_max = [sum(run[i] for run in all_ea_max) / n_runs for i in range(num_ea_points)]
        avg_ea_avg = [sum(run[i] for run in all_ea_avg) / n_runs for i in range(num_ea_points)]
        avg_random_max = [sum(run[i] for run in all_random_max) / n_runs for i in range(num_random_points)]
        avg_random_avg = [sum(run[i] for run in all_random_avg) / n_runs for i in range(num_random_points)]
        
        with open(log_filepath, "w") as log_file:
            log_file.write("Time,EA Max Fitness,EA Avg Fitness,Random Max Fitness,Random Avg Fitness\n")
            for i in range(num_random_points):
                log_file.write(f"{avg_random_x[i]:.2f},,,{avg_random_max[i]:.6f},{avg_random_avg[i]:.6f}\n")
            for i in range(num_ea_points):
                log_file.write(f"{avg_ea_x[i]:.2f},{avg_ea_max[i]:.6f},{avg_ea_avg[i]:.6f},,\n")
        
        plot_timed_results(timestamp, algorithm_type, avg_ea_max, avg_ea_avg, avg_random_max, avg_random_avg, avg_ea_x, avg_random_x)
    
    print("Optimisation complete. Averaged results saved in:", log_filepath)

# Plotting Logic
def plot_iteration_results(timestamp, algorithm_type, ea_max, ea_avg, random_max, random_avg, x_values):
    plt.clf()
    plt.plot(x_values, ea_max, label="EA Max Fitness", color="blue")
    plt.plot(x_values, ea_avg, label="EA Avg Fitness", color="blue", linestyle="--")
    plt.plot(x_values, random_max, label="Random Max Overall Fitness", color="orange")
    plt.plot(x_values, random_avg, label="Random Avg Fitness", color="orange", linestyle="--")
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.ylim(0, 1)
    plt.title("Fitness Over Iterations (Averaged over 10 runs)")
    plt.grid(True)
    plt.savefig(f"Experiment Results/Charts/{algorithm_type.__name__.capitalize()}/Iteration_Execution/{timestamp}.png")
    plt.close()

def plot_timed_results(timestamp, algorithm_type, ea_max, ea_avg, random_max, random_avg, ea_x_values, random_x_values):
    plt.clf()
    plt.plot(random_x_values, random_max, label="Random Max Overall Fitness", color="orange")
    plt.plot(random_x_values, random_avg, label="Random Avg Fitness", color="orange", linestyle="--")
    plt.plot(ea_x_values, ea_max, label="EA Max Fitness", color="blue")
    plt.plot(ea_x_values, ea_avg, label="EA Avg Fitness", color="blue", linestyle="--")
    plt.legend()
    plt.xlabel("Time (seconds)")
    plt.ylabel("Fitness")
    plt.ylim(0, 1)
    combined_x = ea_x_values + random_x_values
    plt.xlim(0, max(combined_x) if combined_x else 1)
    plt.title("Fitness Over Time (Averaged over 10 runs)")
    plt.grid(True)
    plt.savefig(f"Experiment Results/Charts/{algorithm_type.__name__.capitalize()}/Timed_Execution/{timestamp}.png")
    plt.close()

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    execute_optimisation(timestamp, simple_optimiser, mode="iteration",
                         population=population, qubits=qubits,
                         initial_circuit_depth=initial_circuit_depth,
                         elitism_number=elitism_number, iterations=20000, n_runs=1)
    
    # For timed execution, uncomment below:
    # execute_optimisation(timestamp, simple_optimiser, mode="timed",
    #                      population=population, qubits=qubits,
    #                      initial_circuit_depth=initial_circuit_depth,
    #                      elitism_number=elitism_number, runtime_minutes=1/3)
