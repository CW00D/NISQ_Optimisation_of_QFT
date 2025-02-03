import time
import os
from datetime import datetime
import matplotlib.pyplot as plt
from simple_optimiser import initialize_chromosomes, evaluate_random_circuits, get_circuits, get_circuit_fitnesses, apply_genetic_operators

def plot_results(timestamp, mode, ea_max, ea_avg, random_max, random_avg, ea_x_values, random_x_values, title):
    plt.plot(random_x_values, random_max, label="Random Max Overall Fitness", color="orange")
    plt.plot(random_x_values, random_avg, label="Random Avg Fitness", color="orange", linestyle="--")
    plt.plot(ea_x_values, ea_max, label="EA Max Fitness", color="blue")
    plt.plot(ea_x_values, ea_avg, label="EA Avg Fitness", color="blue", linestyle="--")
    plt.legend()
    plt.xlabel("Time (seconds)")
    plt.ylabel("Fitness")
    plt.ylim(0, 1)
    plt.xlim(0, (len(ea_x_values) > 0 and max(ea_x_values)) or (len(random_x_values) > 0 and max(random_x_values)) or 1)
    plt.title(title)
    plt.grid(True)
    plt.savefig(f"Experiment Results/Charts/{mode.capitalize()}_Execution_" + timestamp + ".png")

def execute_optimisation(timestamp, mode="iteration", iterations=1000, runtime_minutes=10, population=20, qubits=3, initial_circuit_depth=10):
    results_folder = "Experiment Results/Logs"
    os.makedirs(results_folder, exist_ok=True)

    log_filename = f"{mode.capitalize()}_Execution_{timestamp}.log"
    log_filepath = os.path.join(results_folder, log_filename)

    random_max_fitnesses, random_avg_fitnesses = [], []
    ea_max_fitnesses, ea_avg_fitnesses = [], []
    max_random_fitness = 0  # Track overall max random fitness
    chromosomes = initialize_chromosomes(population, qubits, initial_circuit_depth)
    
    with open(log_filepath, "w") as log_file:
        log_file.write("Time/Iteration,EA Max Fitness,EA Avg Fitness,Random Max Overall Fitness,Random Avg Fitness\n")

        if mode == "iteration":
            x_values = list(range(iterations))
            for i in x_values:
                random_max_fitness, random_avg_fitness = evaluate_random_circuits(population, 1, qubits, initial_circuit_depth)
                max_random_fitness = max(max_random_fitness, random_max_fitness)
                random_max_fitnesses.append(max_random_fitness)
                random_avg_fitnesses.append(random_avg_fitness)
                
                circuits = get_circuits(chromosomes)
                fitnesses = get_circuit_fitnesses(circuits, qubits)
                max_fitness, avg_fitness = max(fitnesses), sum(fitnesses) / len(fitnesses)
                ea_max_fitnesses.append(max_fitness)
                ea_avg_fitnesses.append(avg_fitness)
                
                log_file.write(f"{i},{max_fitness:.6f},{avg_fitness:.6f},{max_random_fitness:.6f},{random_avg_fitness:.6f}\n")
                
                if max_fitness >= 1:
                    print("Stopping early: Fitness threshold reached")
                    break

                chromosomes = apply_genetic_operators(chromosomes, fitnesses, population // 4)
            
            title = "Fitness Over Iterations"
            x_label = "Iterations"
        
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
                random_max_fitness, random_avg_fitness = evaluate_random_circuits(population, 1, qubits, initial_circuit_depth)
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
                circuits = get_circuits(chromosomes)
                fitnesses = get_circuit_fitnesses(circuits, qubits)
                max_fitness, avg_fitness = max(fitnesses), sum(fitnesses) / len(fitnesses)
                
                if elapsed_time >= (log_interval * values_logged_ea):
                    ea_x_values.append(elapsed_time)
                    ea_max_fitnesses.append(max_fitness)
                    ea_avg_fitnesses.append(avg_fitness)
                    log_file.write(f"{elapsed_time:.2f},{max_fitness:.6f},{avg_fitness:.6f},,\n")
                    values_logged_ea += 1
                
                chromosomes = apply_genetic_operators(chromosomes, fitnesses, population // 4)
            
            title = "Fitness Over Time"
    
    plot_results(timestamp, mode, ea_max_fitnesses, ea_avg_fitnesses, random_max_fitnesses, random_avg_fitnesses, ea_x_values, random_x_values, title)
    print("Optimisation complete. Results saved in:", log_filepath)

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    execute_optimisation(timestamp, mode="iteration", iterations=100)
    #execute_optimisation(timestamp, mode="timed", runtime_minutes=1)
