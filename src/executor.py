import time
import os
import random
import numpy as np
import concurrent.futures
import matplotlib
matplotlib.use("Agg")
from qiskit.quantum_info import Statevector

# ================================
# Simulator Selection
# ================================
#import optimiser_simple as optimiser #10|10
import optimiser_depth_reduction as optimiser #10|2
#import optimiser_noisy as optimiser #10|10
#import optimiser_noisy_depth_reduction as optimiser #10|10

# ================================
# Global Execution Parameters
# ================================
qubits = 3

BASELINE_ITERATIONS = 2000
N_RANDOM_RUNS = 2
RANDOM_BASELINE_DIR = "Experiment Results/Random_Baseline"  
RANDOM_BASELINE_FILE = os.path.join(RANDOM_BASELINE_DIR, "random_baseline_" + str(qubits) + "_qubit_simple.csv")

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
            print(i)
            random_max_fitness, random_avg_fitness = optimiser.evaluate_random_circuits(1, qubits, target_states)
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
# Saving Intermediate Results
# ================================
def save_intermediate_chromosomes(iteration, sorted_chromosomes, chromosome_filepath):
    # Always overwrite the file so that only the latest top 10 chromosomes are saved.
    with open(chromosome_filepath, "w") as cf:
        cf.write("Iteration,Rank,Fitness,Chromosome\n")
        for idx, (fitness, chromosome) in enumerate(sorted_chromosomes[:10]):
            cf.write(f"{iteration},{idx},{fitness:.6f},\"{chromosome}\"\n")
    print(f"Overwritten intermediate chromosomes at iteration {iteration} to {chromosome_filepath}")

def save_fitness_values(iteration, fitnesses, fitness_csv_filepath, run):
    """
    Save the full list of fitness values for the current iteration.
    The CSV will have columns: Iteration, Run, ChromosomeIndex, Fitness.
    Appends new rows (writing header only if file does not exist).
    """
    write_header = not os.path.exists(fitness_csv_filepath)
    mode = "a"  # append mode
    with open(fitness_csv_filepath, mode) as f:
        if write_header:
            f.write("Iteration,Run,ChromosomeIndex,Fitness\n")
        for idx, fit in enumerate(fitnesses):
            f.write(f"{iteration},{run},{idx},{fit:.6f}\n")

# ================================
# Combined Single EA Run (Parallel Task)
# ================================
def run_single_run(run, iterations, qubits, target_states, intermediate_chromosome_filepath, fitness_csv_filepath, final_chromosome_filepath):
    # Set a unique seed per run for reproducibility.
    seed_value = int(time.time()) + run
    random.seed(seed_value)
    np.random.seed(seed_value)
    
    chromosomes = optimiser.initialize_chromosomes(qubits)
    run_ea_max = []
    run_ea_avg = []
    
    for i in range(iterations):
        if i % 100 == 0:
            print(f"Run {run}, iteration {i}")
        circuits = optimiser.get_circuits(chromosomes)

        # Get full fitness values for all individuals
        fitnesses = optimiser.get_circuit_fitnesses(target_states, circuits, chromosomes)
        max_fitness = max(fitnesses)
        avg_fitness = sum(fitnesses) / len(fitnesses)
        run_ea_max.append(max_fitness)
        run_ea_avg.append(avg_fitness)

        # Save top 10 chromosomes every 100 iterations (and at the final iteration)
        if i % 100 == 0 or i == iterations - 1:
            sorted_chromosomes = sorted(zip(fitnesses, chromosomes), key=lambda x: x[0], reverse=True)
            save_intermediate_chromosomes(i, sorted_chromosomes, intermediate_chromosome_filepath)
            save_fitness_values(i, fitnesses, fitness_csv_filepath, run)
        
        chromosomes = optimiser.apply_genetic_operators(chromosomes, fitnesses)

    final_circuits = optimiser.get_circuits(chromosomes)
    final_fitnesses = optimiser.get_circuit_fitnesses(target_states, final_circuits, chromosomes)
    sorted_final = sorted(zip(final_fitnesses, chromosomes), key=lambda x: x[0], reverse=True)
    
    # Save the final top 10 chromosomes for this run
    with open(final_chromosome_filepath, "w") as cf:
        cf.write("Rank,Fitness,Chromosome\n")
        for idx, (fitness, chromosome) in enumerate(sorted_final[:10]):
            cf.write(f"{idx},{fitness:.6f},\"{chromosome}\"\n")
    print(f"Top chromosomes for run {run} saved in:", final_chromosome_filepath)
    
    return run_ea_max, run_ea_avg, sorted_final

# ================================
# Combined Parallel EA Execution
# ================================
def execute_optimisation(simulation_name, iterations, n_runs=10):
    # Create a folder for the specific simulation type (optimiser name)
    sim_type_folder = os.path.join("Experiment Results", optimiser.__name__.capitalize())
    os.makedirs(sim_type_folder, exist_ok=True)
    
    # Create a simulation folder inside the simulation type folder using the simulation name.
    simulation_folder = os.path.join(sim_type_folder, simulation_name)
    os.makedirs(simulation_folder, exist_ok=True)
    
    # Create a Data folder inside the simulation folder.
    data_folder = os.path.join(simulation_folder, "Data")
    os.makedirs(data_folder, exist_ok=True)
    
    # Determine the next run number
    existing_files = os.listdir(data_folder)
    run_numbers = [int(f.split('_')[0][3:]) for f in existing_files if f.startswith('run') and f.split('_')[0][3:].isdigit()]
    run_number = max(run_numbers, default=0) + 1
    
    # Define file paths within the Data folder.
    iteration_data_filepath = os.path.join(data_folder, f"run{run_number}_iteration_data.csv")
    intermediate_filepaths = [
        os.path.join(data_folder, f"run{run_number}_intermediate.csv")
        for run in range(n_runs)
    ]
    
    fitness_filepaths = [
        os.path.join(data_folder, f"run{run_number}_fitness.csv")
        for run in range(n_runs)
    ]
    final_chromosome_filepaths = [
        os.path.join(data_folder, f"run{run_number}_final_chromosomes.csv")
        for run in range(n_runs)
    ]
    
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
                executor.submit(run_single_run, run, iterations, qubits, target_states, intermediate_filepaths[run], fitness_filepaths[run], final_chromosome_filepaths[run])
            )
        for future in concurrent.futures.as_completed(futures):
            run_ea_max, run_ea_avg, sorted_final = future.result()
            all_ea_max.append(run_ea_max)
            all_ea_avg.append(run_ea_avg)
            overall_final.extend(sorted_final)
    
    # Average the EA results across runs.
    avg_ea_max = [sum(run[i] for run in all_ea_max) / n_runs for i in range(iterations)]
    avg_ea_avg = [sum(run[i] for run in all_ea_avg) / n_runs for i in range(iterations)]
    
    # Ensure baseline data is long enough
    if len(baseline_max) < iterations:
        baseline_max.extend([baseline_max[-1]] * (iterations - len(baseline_max)))
    if len(baseline_avg) < iterations:
        baseline_avg.extend([baseline_avg[-1]] * (iterations - len(baseline_avg)))
    
    # Write the iteration data to a CSV file.
    with open(iteration_data_filepath, "w") as iteration_data_file:
        iteration_data_file.write("Iteration,EA Max Fitness,EA Avg Fitness,Random Max Fitness,Random Avg Fitness\n")
        for i in range(iterations):
            iteration_data_file.write(f"{i},{avg_ea_max[i]:.6f},{avg_ea_avg[i]:.6f},{baseline_max[i]:.6f},{baseline_avg[i]:.6f}\n")
    
    print("Optimisation complete. Averaged results saved in:", iteration_data_filepath)
    
    # Display the top 10 circuits overall.
    overall_final_sorted = sorted(overall_final, key=lambda x: x[0], reverse=True)
    print("\n" + "="*30)
    print("Top 10 circuits overall from all runs:")
    for idx, (fitness, chromosome) in enumerate(overall_final_sorted[:10]):
        circuit = optimiser.get_circuits([chromosome])[0]
        print(f"\nCircuit {idx} (Fitness: {fitness:.6f}):")
        print(circuit.draw())

# ================================
# Main Execution Block
# ================================
if __name__ == "__main__":
    simulation_name = str(qubits) + " Qubit Simulation"  # Specify the simulation name
    # Adjust iterations and n_runs as needed.
    if qubits == 2:
        execute_optimisation(simulation_name, iterations=2000, n_runs=1)
    if qubits == 3:
        execute_optimisation(simulation_name, iterations=20000, n_runs=1)