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
import optimiser_simple as optimiser
#import optimiser_noisy as optimiser
#import optimiser_depth_reduction as optimiser


# ================================
# Global Execution Parameters
# ================================
population = 100
#population = 10
qubits = 3
initial_circuit_depth = 10

# Mutation Parameters
elitism_number = population // 3  
parameter_mutation_rate = 0.1
gate_mutation_rate = 0.3
layer_mutation_rate = 0.2
max_parameter_mutation = 0.2
layer_deletion_rate = 0.03

# ================================
# Helper Functions
# ================================
def save_intermediate_chromosomes(iteration, sorted_chromosomes, chromosome_filepath):
    """Saves the top 10 chromosomes at a given iteration to a CSV file (overwriting each time)."""
    with open(chromosome_filepath, "w") as cf:
        cf.write("Iteration,Rank,Fitness,Chromosome\n")
        for idx, (fitness, chromosome) in enumerate(sorted_chromosomes[:10]):
            cf.write(f"{iteration},{idx},{fitness:.6f},\"{chromosome}\"\n")
    print(f"Saved intermediate chromosomes at iteration {iteration} to {chromosome_filepath}")

# ================================
# Single EA Run (Parallel Task)
# ================================
def run_single_run(run, iterations, population, qubits, initial_circuit_depth,
                   initial_states, target_states, elitism_number, parameter_mutation_rate,
                   gate_mutation_rate, layer_mutation_rate, max_parameter_mutation, layer_deletion_rate,
                   chromosome_filepath):

    seed_value = int(time.time()) + run
    random.seed(seed_value)
    np.random.seed(seed_value)
    
    chromosomes = optimiser.initialize_chromosomes(population, qubits, initial_circuit_depth)

    for i in range(iterations):
        if i % 100 == 0:
            print(f"Run {run}, iteration {i}")

        circuits = optimiser.get_circuits(chromosomes)
        fitnesses = optimiser.get_circuit_fitnesses(target_states, circuits, chromosomes)

        # Store top 10 chromosomes every 100 iterations
        if i % 100 == 0 or i == iterations - 1:
            sorted_chromosomes = sorted(zip(fitnesses, chromosomes), key=lambda x: x[0], reverse=True)
            save_intermediate_chromosomes(i, sorted_chromosomes, chromosome_filepath)

        # Early stopping if max fitness is reached
        if max(fitnesses) >= 1:
            break

        # Apply genetic operators
        chromosomes = optimiser.apply_genetic_operators(
            chromosomes, fitnesses, elitism_number, parameter_mutation_rate,
            gate_mutation_rate, layer_mutation_rate, max_parameter_mutation, layer_deletion_rate
        )

    return sorted(zip(fitnesses, chromosomes), key=lambda x: x[0], reverse=True)

# ================================
# Parallel EA Execution
# ================================
def execute_optimisation(timestamp, iterations, n_runs=1):
    chromosome_folder = f"Experiment Results/Chromosomes/{optimiser.__name__.capitalize()}"
    os.makedirs(chromosome_folder, exist_ok=True)
    chromosome_filepath = os.path.join(chromosome_folder, f"{timestamp}.csv")

    initial_states = [Statevector.from_label(f"{i:0{qubits}b}") for i in range(2**qubits)]
    target_states = optimiser.get_qft_target_states(qubits)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                run_single_run, run, iterations, population, qubits,
                initial_circuit_depth, initial_states, target_states, elitism_number,
                parameter_mutation_rate, gate_mutation_rate, layer_mutation_rate,
                max_parameter_mutation, layer_deletion_rate, chromosome_filepath
            )
            for run in range(n_runs)
        ]

        for future in concurrent.futures.as_completed(futures):
            final_chromosomes = future.result()

    print("Final top chromosomes saved at:", chromosome_filepath)

# ================================
# Main Execution Block
# ================================
if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    execute_optimisation(timestamp, iterations=20000, n_runs=1)
