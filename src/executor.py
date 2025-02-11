import time
import os
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import simple_optimiser
#import simple_optimiser_noisy
#import moea_optimiser
#import moea_optimiser_noisy

import random
import numpy as np

# Simulation Parameters
population = 20
qubits = 3
initial_circuit_depth = 10

# Mutation Parameters
elitism_number = population // 3  # e.g. 3
parameter_mutation_rate = 0.2       # 0.2
gate_mutation_rate = 0.4            # 0.4
layer_mutation_rate = 0.3           # 0.3
max_parameter_mutation = 0.3        # 0.3
layer_deletion_rate = 0.05          # 0.05

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
    # Using forward slashes for cross-platform compatibility:
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

def execute_optimisation(timestamp, algorithm_type, mode, population, qubits, initial_circuit_depth, elitism_number, iterations=-1, runtime_minutes=-1):
    # Create a folder to save log files
    results_folder = f"Experiment Results/Logs/{algorithm_type.__name__.capitalize()}/{mode.capitalize()}_Execution"
    os.makedirs(results_folder, exist_ok=True)
    log_filename = f"{timestamp}.log"
    log_filepath = os.path.join(results_folder, log_filename)
    
    n_runs = 10  # Number of independent runs to average over

    if mode == "iteration":
        # Prepare lists to collect the results from each run.
        all_ea_max = []
        all_ea_avg = []
        all_random_max = []
        all_random_avg = []
        for run in range(n_runs):
            # Set a unique random seed for each run:
            seed_value = int(time.time()) + run
            random.seed(seed_value)
            np.random.seed(seed_value)
            
            # (Re)initialize chromosomes and target states for this run.
            chromosomes = algorithm_type.initialize_chromosomes(population, qubits, initial_circuit_depth)
            target_states = algorithm_type.get_qft_target_states(qubits)
            run_ea_max = []
            run_ea_avg = []
            run_random_max = []
            run_random_avg = []
            current_random_max = 0  # For maintaining the running maximum for random circuits
            
            for i in range(iterations):
                # Evaluate random circuits.
                random_max_fitness, random_avg_fitness = algorithm_type.evaluate_random_circuits(population, 1, qubits, initial_circuit_depth, target_states)
                current_random_max = max(current_random_max, random_max_fitness)
                run_random_max.append(current_random_max)
                run_random_avg.append(random_avg_fitness)
                
                # Evaluate EA circuits.
                circuits = algorithm_type.get_circuits(chromosomes)
                fitnesses = algorithm_type.get_circuit_fitnesses(target_states, circuits, qubits)
                max_fitness = max(fitnesses)
                avg_fitness = sum(fitnesses) / len(fitnesses)
                run_ea_max.append(max_fitness)
                run_ea_avg.append(avg_fitness)
                
                # Stop early if a perfect circuit is found; pad the rest of the run with the current values.
                if max_fitness >= 1:
                    remaining = iterations - i - 1
                    run_ea_max.extend([max_fitness] * remaining)
                    run_ea_avg.extend([avg_fitness] * remaining)
                    run_random_max.extend([current_random_max] * remaining)
                    run_random_avg.extend([random_avg_fitness] * remaining)
                    break
                
                # Apply genetic operators.
                chromosomes = algorithm_type.apply_genetic_operators(
                    chromosomes, fitnesses, elitism_number, 
                    parameter_mutation_rate, gate_mutation_rate, 
                    layer_mutation_rate, max_parameter_mutation, 
                    layer_deletion_rate
                )
            
            # Save the results from this run.
            all_ea_max.append(run_ea_max)
            all_ea_avg.append(run_ea_avg)
            all_random_max.append(run_random_max)
            all_random_avg.append(run_random_avg)
        
        # Compute the elementwise average across all runs.
        avg_ea_max = [sum(run[i] for run in all_ea_max) / n_runs for i in range(iterations)]
        avg_ea_avg = [sum(run[i] for run in all_ea_avg) / n_runs for i in range(iterations)]
        avg_random_max = [sum(run[i] for run in all_random_max) / n_runs for i in range(iterations)]
        avg_random_avg = [sum(run[i] for run in all_random_avg) / n_runs for i in range(iterations)]
        x_values = list(range(iterations))
        
        # Write the averaged values to the log file.
        with open(log_filepath, "w") as log_file:
            log_file.write("Iteration,EA Max Fitness,EA Avg Fitness,Random Max Fitness,Random Avg Fitness\n")
            for i in range(iterations):
                log_file.write(f"{i},{avg_ea_max[i]:.6f},{avg_ea_avg[i]:.6f},{avg_random_max[i]:.6f},{avg_random_avg[i]:.6f}\n")
        
        plot_iteration_results(timestamp, algorithm_type, avg_ea_max, avg_ea_avg, avg_random_max, avg_random_avg, x_values)
    
    elif mode == "timed":
        # Prepare lists to collect the results from each run.
        all_ea_max = []
        all_ea_avg = []
        all_random_max = []
        all_random_avg = []
        all_ea_x = []
        all_random_x = []
        for run in range(n_runs):
            seed_value = int(time.time()) + run
            random.seed(seed_value)
            np.random.seed(seed_value)
            
            chromosomes = algorithm_type.initialize_chromosomes(population, qubits, initial_circuit_depth)
            target_states = algorithm_type.get_qft_target_states(qubits)
            
            # Run the random circuit evaluation for the first half of the runtime.
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
                    population, 1, qubits, initial_circuit_depth, target_states
                )
                current_random_max = max(current_random_max, random_max_fitness)
                if elapsed_time >= (log_interval * values_logged_random):
                    run_random_x.append(elapsed_time)
                    run_random_max.append(current_random_max)
                    run_random_avg.append(random_avg_fitness)
                    values_logged_random += 1
            
            # Run the EA for the second half of the runtime.
            ea_start_time = time.time()
            run_ea_max = []
            run_ea_avg = []
            run_ea_x = []
            values_logged_ea = 0
            elapsed_time_ea = 0
            while elapsed_time_ea < half_runtime:
                elapsed_time_ea = time.time() - ea_start_time
                circuits = algorithm_type.get_circuits(chromosomes)
                fitnesses = algorithm_type.get_circuit_fitnesses(target_states, circuits, qubits)
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
        
        # For averaging we assume that all runs have logged the same number of points.
        num_random_points = len(all_random_x[0])
        num_ea_points = len(all_ea_x[0])
        avg_random_x = all_random_x[0]  # (assumed identical across runs)
        avg_ea_x = all_ea_x[0]          # (assumed identical across runs)
        
        avg_ea_max = [sum(run[i] for run in all_ea_max) / n_runs for i in range(num_ea_points)]
        avg_ea_avg = [sum(run[i] for run in all_ea_avg) / n_runs for i in range(num_ea_points)]
        avg_random_max = [sum(run[i] for run in all_random_max) / n_runs for i in range(num_random_points)]
        avg_random_avg = [sum(run[i] for run in all_random_avg) / n_runs for i in range(num_random_points)]
        
        # Write the averaged values to the log file.
        with open(log_filepath, "w") as log_file:
            log_file.write("Time,EA Max Fitness,EA Avg Fitness,Random Max Fitness,Random Avg Fitness\n")
            # Log the random part.
            for i in range(num_random_points):
                log_file.write(f"{avg_random_x[i]:.2f},,,{avg_random_max[i]:.6f},{avg_random_avg[i]:.6f}\n")
            # Then log the EA part.
            for i in range(num_ea_points):
                log_file.write(f"{avg_ea_x[i]:.2f},{avg_ea_max[i]:.6f},{avg_ea_avg[i]:.6f},,\n")
        
        plot_timed_results(timestamp, algorithm_type, avg_ea_max, avg_ea_avg, avg_random_max, avg_random_avg, avg_ea_x, avg_random_x)
    
    print("Optimisation complete. Averaged results saved in:", log_filepath)

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    # To run in iteration mode (for example, 100 iterations):
    execute_optimisation(timestamp, simple_optimiser, mode="iteration",
                         population=population, qubits=qubits,
                         initial_circuit_depth=initial_circuit_depth,
                         elitism_number=elitism_number, iterations=1000)
    
    # To run in timed mode (for example, for 20 seconds = 1/3 minute):
    #execute_optimisation(timestamp, simple_optimiser, mode="timed",
    #                     population=population, qubits=qubits,
    #                     initial_circuit_depth=initial_circuit_depth,
    #                     elitism_number=elitism_number, runtime_minutes=1/3)
