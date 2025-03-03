# parameter_exploration.py
import os
import time
import csv
from datetime import datetime
import executor
import optimiser_simple

# === Define the grid for the mutation parameters ===
# (Adjust these ranges/values as needed for your exploration.)
parameter_mutation_rates = [0.1]
gate_mutation_rates = [0.3]
layer_mutation_rates = [0.2, 0.3]
max_parameter_mutations = [0.1, 0.2, 0.3, 0.4]
layer_deletion_rates = [0.03]

# Number of iterations for each experiment run.
iterations = 50

# Dictionaries to store results.
performance_results = {}  # Mapping parameter-set string -> final EA max fitness
best_perf = -1.0
best_params = None

# Loop over every combination of mutation parameters.
for pmr in parameter_mutation_rates:
    for gmr in gate_mutation_rates:
        for lmr in layer_mutation_rates:
            for mpm in max_parameter_mutations:
                for ldr in layer_deletion_rates:
                    # Update the executor's global mutation parameters.
                    executor.parameter_mutation_rate = pmr
                    executor.gate_mutation_rate = gmr
                    executor.layer_mutation_rate = lmr
                    executor.max_parameter_mutation = mpm
                    executor.layer_deletion_rate = ldr

                    # Create a string that identifies this parameter set.
                    param_str = f"pmr{pmr}_gmr{gmr}_lmr{lmr}_mpm{mpm}_ldr{ldr}"
                    
                    # Create a unique timestamp that includes the parameter info.
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    full_timestamp = f"{timestamp}_{param_str}"
                    
                    print(f"Running experiment with parameters: {param_str}")
                    
                    # Run the optimisation experiment (in iteration mode).
                    # Note: We use the globals from executor for population, qubits, etc.
                    executor.execute_optimisation(
                        full_timestamp,
                        optimiser_simple,
                        mode="iteration",
                        population=executor.population,
                        qubits=executor.qubits,
                        initial_circuit_depth=executor.initial_circuit_depth,
                        elitism_number=executor.elitism_number,
                        iterations=iterations
                    )
                    
                    # The log file is written to:
                    #   Experiment Results/Logs/{Algorithm}/Iteration_Execution/{full_timestamp}.log
                    log_folder = os.path.join("Experiment Results", "Logs",
                                              optimiser_simple.__name__.capitalize(),
                                              "Iteration_Execution")
                    log_file_path = os.path.join(log_folder, f"{full_timestamp}.log")
                    
                    # (Give a short pause if needed for file I/O.)
                    time.sleep(1)
                    
                    if os.path.exists(log_file_path):
                        with open(log_file_path, "r") as log_file:
                            lines = log_file.readlines()
                            if len(lines) > 1:
                                # The first line is a header; use the last line for the final performance.
                                last_line = lines[-1]
                                # Expected format: Iteration,EA Max Fitness,EA Avg Fitness,Random Max Fitness,Random Avg Fitness
                                parts = last_line.strip().split(",")
                                try:
                                    final_ea_max = float(parts[1])
                                except Exception as e:
                                    print(f"Error parsing log file {log_file_path}: {e}")
                                    final_ea_max = 0.0
                                performance_results[param_str] = final_ea_max
                                print(f"Final EA Max Fitness for {param_str}: {final_ea_max}")
                                if final_ea_max > best_perf:
                                    best_perf = final_ea_max
                                    best_params = param_str
                            else:
                                print(f"Log file {log_file_path} does not contain data.")
                    else:
                        print(f"Log file {log_file_path} was not found.")
                    
                    # Pause a bit between experiments to ensure unique timestamps.
                    time.sleep(1)

# Summarise the exploration.
print("\nParameter exploration complete.")
if best_params is not None:
    print("Best parameters:", best_params, "with final EA max fitness:", best_perf)
else:
    print("No valid results were obtained.")

# Optionally, write the full results to a CSV file.
csv_filename = "parameter_exploration_results.csv"
with open(csv_filename, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Parameter Set", "Final EA Max Fitness"])
    for param_set, fitness in performance_results.items():
        writer.writerow([param_set, fitness])
print("Results written to", csv_filename)
