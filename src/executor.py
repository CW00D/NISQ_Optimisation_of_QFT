import time
import os
import random
import numpy as np
from datetime import datetime
import concurrent.futures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qiskit.quantum_info import Statevector

# ================================
# Simulator Selection
# ================================
#import optimiser_simple as optimiser
import optimiser_noisy as optimiser
#import optimiser_depth_reduction as optimiser
#import optimiser_noisy_depth_reduction as optimiser

# ================================
# Global Execution Parameters
# ================================
#population = 100
population = 1
qubits = 3
initial_circuit_depth = 10

# Mutation Parameters
elitism_number = population // 3      
parameter_mutation_rate = 0.1
gate_mutation_rate = 0.3
layer_mutation_rate = 0.2
max_parameter_mutation = 0.2
layer_deletion_rate = 0.03

# Random Baseline Parameters
BASELINE_ITERATIONS = 20000
N_RANDOM_RUNS = 10
RANDOM_BASELINE_DIR = "Experiment Results/Random_Baseline"
RANDOM_BASELINE_FILE = os.path.join(RANDOM_BASELINE_DIR, "random_baseline.csv")

# ================================
# Random Baseline Functions
# ================================
def compute_random_baseline(baseline_iterations=BASELINE_ITERATIONS, n_random_runs=N_RANDOM_RUNS):
    target_states = optimiser.get_qft_target_states(qubits)
    
    all_random_max = []
    all_random_avg = []
    for run in range(n_random_runs):
        print(f"Random baseline run {run}")
        current_random_max = 0
        run_random_max = []
        run_random_avg = []
        for i in range(baseline_iterations):
            random_max_fitness, random_avg_fitness = optimiser.evaluate_random_circuits(
                population, 1, qubits, initial_circuit_depth, target_states
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

def ensure_random_baseline(baseline_iterations=BASELINE_ITERATIONS, n_random_runs=N_RANDOM_RUNS):
    if os.path.exists(RANDOM_BASELINE_FILE):
        print("Loading existing random baseline...")
        baseline_max, baseline_avg = load_random_baseline(baseline_iterations)
        if baseline_max is not None:
            return baseline_max, baseline_avg
    print("Computing random baseline...")
    baseline_max, baseline_avg = compute_random_baseline(baseline_iterations, n_random_runs)
    save_random_baseline(baseline_max, baseline_avg)
    return baseline_max, baseline_avg

# ================================
# Chart Plotting Logic
# ================================
def plot_iteration_results(timestamp, ea_max, ea_avg, random_max, random_avg, x_values, run_count, iteration_count, output_dir):
    plt.clf()
    plt.plot(x_values, ea_max, label="EA Max Fitness", color="blue")
    plt.plot(x_values, ea_avg, label="EA Avg Fitness", color="blue", linestyle="--")
    plt.plot(x_values, random_max, label="Random Max Overall Fitness", color="orange")
    plt.plot(x_values, random_avg, label="Random Avg Fitness", color="orange", linestyle="--")
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2)
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.ylim(0, 1)
    plt.title(f"Fitness Over {iteration_count} Iterations (Avg Over {run_count} Runs)")
    plt.grid(True)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{timestamp}.png"), bbox_inches='tight')
    plt.close()

# ================================
# Intermediate Chromosome Saving
# ================================
def save_intermediate_chromosomes(iteration, sorted_chromosomes, chromosome_filepath):
    mode = "a" if os.path.exists(chromosome_filepath) else "w"
    with open(chromosome_filepath, mode) as cf:
        if mode == "w":
            cf.write("Iteration,Rank,Fitness,Chromosome\n")
        for idx, (fitness, chromosome) in enumerate(sorted_chromosomes[:10]):
            cf.write(f"{iteration},{idx},{fitness:.6f},\"{chromosome}\"\n")
    print(f"Saved intermediate chromosomes at iteration {iteration} to {chromosome_filepath}")

# ================================
# Combined Single EA Run (Parallel Task)
# ================================
def run_single_run(run, iterations, population, qubits, initial_circuit_depth,
                   initial_states, target_states, elitism_number, parameter_mutation_rate,
                   gate_mutation_rate, layer_mutation_rate, max_parameter_mutation, layer_deletion_rate,
                   intermediate_chromosome_filepath):
    # Set a unique seed per run for reproducibility.
    seed_value = int(time.time()) + run
    random.seed(seed_value)
    np.random.seed(seed_value)
    
    chromosomes = optimiser.initialize_chromosomes(population, qubits, initial_circuit_depth)
    run_ea_max = []
    run_ea_avg = []
    
    for i in range(iterations):
        if i % 100 == 0:
            print(f"Run {run}, iteration {i}")
        circuits = optimiser.get_circuits(chromosomes)
        fitnesses = optimiser.get_circuit_fitnesses(target_states, circuits, chromosomes)
        max_fitness = max(fitnesses)
        avg_fitness = sum(fitnesses) / len(fitnesses)
        run_ea_max.append(max_fitness)
        run_ea_avg.append(avg_fitness)
        
        # Save top 10 chromosomes every 100 iterations (and at the final iteration)
        if i % 100 == 0 or i == iterations - 1:
            sorted_chromosomes = sorted(zip(fitnesses, chromosomes), key=lambda x: x[0], reverse=True)
            save_intermediate_chromosomes(i, sorted_chromosomes, intermediate_chromosome_filepath)
        
        # Early stopping if optimum fitness is reached.
        if max_fitness >= 1:
            remaining = iterations - i - 1
            run_ea_max.extend([max_fitness] * remaining)
            run_ea_avg.extend([avg_fitness] * remaining)
            break
        
        chromosomes = optimiser.apply_genetic_operators(
            chromosomes, fitnesses, elitism_number, parameter_mutation_rate,
            gate_mutation_rate, layer_mutation_rate, max_parameter_mutation, layer_deletion_rate
        )
    
    final_circuits = optimiser.get_circuits(chromosomes)
    final_fitnesses = optimiser.get_circuit_fitnesses(target_states, final_circuits, chromosomes)
    sorted_final = sorted(zip(final_fitnesses, chromosomes), key=lambda x: x[0], reverse=True)
    return run_ea_max, run_ea_avg, sorted_final

# ================================
# Combined Parallel EA Execution
# ================================
def execute_optimisation(timestamp, iterations, n_runs=10):
    # Create a folder for the specific simulation type (optimiser name)
    sim_type_folder = os.path.join("Experiment Results", optimiser.__name__.capitalize())
    os.makedirs(sim_type_folder, exist_ok=True)
    
    # Create a simulation folder inside the simulation type folder using the timestamp.
    simulation_folder = os.path.join(sim_type_folder, f"Simulation_{timestamp}")
    os.makedirs(simulation_folder, exist_ok=True)
    
    # Define file paths within the simulation folder.
    log_filepath = os.path.join(simulation_folder, f"{timestamp}.log")
    chart_output_dir = simulation_folder  # Chart will be saved here.
    intermediate_filepaths = [
        os.path.join(simulation_folder, f"{timestamp}_run{run}_intermediate.csv")
        for run in range(n_runs)
    ]
    final_chromosome_filepath = os.path.join(simulation_folder, f"{timestamp}_final_chromosomes.csv")
    
    # Prepare initial and target states.
    initial_states = [Statevector.from_label(f"{i:0{qubits}b}") for i in range(2**qubits)]
    target_states = optimiser.get_qft_target_states(qubits)
    
    # Compute (or load) the random baseline.
    baseline_max, baseline_avg = ensure_random_baseline()
    
    all_ea_max = []
    all_ea_avg = []
    overall_final = []
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for run in range(n_runs):
            futures.append(
                executor.submit(
                    run_single_run, run, iterations, population, qubits,
                    initial_circuit_depth, initial_states, target_states, elitism_number,
                    parameter_mutation_rate, gate_mutation_rate, layer_mutation_rate,
                    max_parameter_mutation, layer_deletion_rate, intermediate_filepaths[run]
                )
            )
        for future in concurrent.futures.as_completed(futures):
            run_ea_max, run_ea_avg, sorted_final = future.result()
            all_ea_max.append(run_ea_max)
            all_ea_avg.append(run_ea_avg)
            overall_final.extend(sorted_final)
    
    # Average the EA results across runs.
    avg_ea_max = [sum(run[i] for run in all_ea_max) / n_runs for i in range(iterations)]
    avg_ea_avg = [sum(run[i] for run in all_ea_avg) / n_runs for i in range(iterations)]
    x_values = list(range(iterations))
    
    # Write the log file.
    with open(log_filepath, "w") as log_file:
        log_file.write("Iteration,EA Max Fitness,EA Avg Fitness,Random Max Fitness,Random Avg Fitness\n")
        for i in range(iterations):
            log_file.write(f"{i},{avg_ea_max[i]:.6f},{avg_ea_avg[i]:.6f},{baseline_max[i]:.6f},{baseline_avg[i]:.6f}\n")
    
    # Plot the results.
    plot_iteration_results(timestamp, avg_ea_max, avg_ea_avg,
                           baseline_max[:iterations], baseline_avg[:iterations],
                           x_values, n_runs, iterations, chart_output_dir)
    print("Optimisation complete. Averaged results saved in:", log_filepath)
    
    # Display the top 10 circuits overall.
    overall_final_sorted = sorted(overall_final, key=lambda x: x[0], reverse=True)
    print("\n" + "="*30)
    print("Top 10 circuits overall from all runs:")
    for idx, (fitness, chromosome) in enumerate(overall_final_sorted[:10]):
        circuit = optimiser.get_circuits([chromosome])[0]
        print(f"\nCircuit {idx} (Fitness: {fitness:.6f}):")
        print(circuit.draw())
    
    # Save the final top 10 chromosomes.
    with open(final_chromosome_filepath, "w") as cf:
        cf.write("Rank,Fitness,Chromosome\n")
        for idx, (fitness, chromosome) in enumerate(overall_final_sorted[:10]):
            cf.write(f"{idx},{fitness:.6f},\"{chromosome}\"\n")
    print("Top chromosomes saved in:", final_chromosome_filepath)

# ================================
# Main Execution Block
# ================================
if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    # Adjust iterations and n_runs as needed.
    execute_optimisation(timestamp, iterations=100, n_runs=1)
