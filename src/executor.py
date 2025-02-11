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

# Simulation Parameters
population=20
qubits=3
initial_circuit_depth=10

# Mutation Parameters
elitism_number = population // 3 # 3
parameter_mutation_rate=0.2 # 0.2
gate_mutation_rate=0.4 # 0.4
layer_mutation_rate=0.3 #0.3
max_parameter_mutation=0.3 # 0.3
layer_deletion_rate=0.05 # 0.05

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
    plt.title("Fitness Over Iterations")
    plt.grid(True)
    plt.savefig(f"Experiment Results\Charts\{algorithm_type.__name__.capitalize()}\Iteration_Execution\\" + timestamp + ".png")
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
    plt.xlim(0, max(ea_x_values + random_x_values) if (ea_x_values + random_x_values) else 1)
    plt.title("Fitness Over Time")
    plt.grid(True)
    plt.savefig(f"Experiment Results\Charts\{algorithm_type.__name__.capitalize()}\Timed_Execution\\" + timestamp + ".png")
    plt.close()

def execute_optimisation(timestamp, algorithm_type, mode, population, qubits, initial_circuit_depth, elitism_number, iterations=-1, runtime_minutes=-1):
    results_folder = f"Experiment Results\Logs\{algorithm_type.__name__.capitalize()}\{mode.capitalize()}_Execution"
    os.makedirs(results_folder, exist_ok=True)

    log_filename = f"{timestamp}.log"
    log_filepath = os.path.join(results_folder, log_filename)

    random_max_fitnesses, random_avg_fitnesses = [], []
    ea_max_fitnesses, ea_avg_fitnesses = [], []
    max_random_fitness = 0  # Track overall max random fitness
    chromosomes = algorithm_type.initialize_chromosomes(population, qubits, initial_circuit_depth)

    target_states = algorithm_type.get_qft_target_states(qubits)
    
    with open(log_filepath, "w") as log_file:
        log_file.write("Time/Iteration,EA Max Fitness,EA Avg Fitness,Random Max Overall Fitness,Random Avg Fitness\n")

        if mode == "iteration":
            x_values = list(range(iterations))
            for i in x_values:
                random_max_fitness, random_avg_fitness = algorithm_type.evaluate_random_circuits(population, 1, qubits, initial_circuit_depth, target_states)
                max_random_fitness = max(max_random_fitness, random_max_fitness)
                random_max_fitnesses.append(max_random_fitness)
                random_avg_fitnesses.append(random_avg_fitness)
                
                circuits = algorithm_type.get_circuits(chromosomes)
                fitnesses = algorithm_type.get_circuit_fitnesses(target_states, circuits, qubits)
                max_fitness, avg_fitness = max(fitnesses), sum(fitnesses) / len(fitnesses)
                ea_max_fitnesses.append(max_fitness)
                ea_avg_fitnesses.append(avg_fitness)
                
                log_file.write(f"{i},{max_fitness:.6f},{avg_fitness:.6f},{max_random_fitness:.6f},{random_avg_fitness:.6f}\n")
                
                if max_fitness >= 1:
                    print("Stopping early: Fitness threshold reached")
                    break

                chromosomes = algorithm_type.apply_genetic_operators(chromosomes, fitnesses, elitism_number, parameter_mutation_rate, gate_mutation_rate, layer_mutation_rate, max_parameter_mutation, layer_deletion_rate)

            plot_iteration_results(timestamp, algorithm_type, ea_max_fitnesses, ea_avg_fitnesses, random_max_fitnesses, random_avg_fitnesses, x_values)
        
        elif mode == "timed":
            start_time = time.time()
            half_runtime = (runtime_minutes * 60) / 2
            log_interval = 10  # Log every 10 seconds
            ea_x_values, random_x_values = [], []
            
            elapsed_time = 0
            values_logged_random = 0
            values_logged_ea = 0

            # Run random generator for half the time
            while elapsed_time < half_runtime:
                elapsed_time = time.time() - start_time
                random_max_fitness, random_avg_fitness = algorithm_type.evaluate_random_circuits(population, 1, qubits, initial_circuit_depth, target_states)
                max_random_fitness = max(max_random_fitness, random_max_fitness)
                
                if elapsed_time >= (log_interval * values_logged_random):
                    random_x_values.append(elapsed_time)
                    random_max_fitnesses.append(max_random_fitness)
                    random_avg_fitnesses.append(random_avg_fitness)
                    log_file.write(f"{elapsed_time:.2f},,,{max_random_fitness:.6f},{random_avg_fitness:.6f}\n")
                    values_logged_random += 1
            
            # Run EA for the second half starting from 0
            ea_start_time = time.time()
            elapsed_time = 0
            while elapsed_time < half_runtime:
                elapsed_time = time.time() - ea_start_time
                circuits = algorithm_type.get_circuits(chromosomes)
                fitnesses = algorithm_type.get_circuit_fitnesses(target_states, circuits, qubits)
                max_fitness, avg_fitness = max(fitnesses), sum(fitnesses) / len(fitnesses)
                
                if elapsed_time >= (log_interval * values_logged_ea):
                    ea_x_values.append(elapsed_time)
                    ea_max_fitnesses.append(max_fitness)
                    ea_avg_fitnesses.append(avg_fitness)
                    log_file.write(f"{elapsed_time:.2f},{max_fitness:.6f},{avg_fitness:.6f},,\n")
                    values_logged_ea += 1
                
                chromosomes = algorithm_type.apply_genetic_operators(chromosomes, fitnesses, elitism_number, parameter_mutation_rate, gate_mutation_rate, layer_mutation_rate, max_parameter_mutation, layer_deletion_rate)

            plot_timed_results(timestamp, algorithm_type, ea_max_fitnesses, ea_avg_fitnesses, random_max_fitnesses, random_avg_fitnesses, ea_x_values, random_x_values)
    
    print("Optimisation complete. Results saved in:", log_filepath)

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    execute_optimisation(timestamp, simple_optimiser, mode="iteration",population=population, elitism_number=elitism_number, qubits=qubits, initial_circuit_depth=initial_circuit_depth, iterations=1)
    #execute_optimisation(timestamp, simple_optimiser, mode="timed", population=population, elitism_number=elitism_number, qubits=qubits, initial_circuit_depth=initial_circuit_depth, runtime_minutes=1/3)
