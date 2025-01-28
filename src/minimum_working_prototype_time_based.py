# Imports
# -----------------------------------------------------
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QFT
from qiskit.quantum_info import Statevector, state_fidelity
import re
import numpy as np
import copy
import matplotlib.pyplot as plt
from datetime import datetime
import os
import time

# Gate lists
# -----------------------------------------------------
single_qubit_gates = [
    "x",
    "y",
    "z",
    "h",
    "s",
    "sdg",
    "t",
    "tdg",
    "rx",
    "ry",
    "rz"
]

double_qubit_gates = [
    "cx",
    "cy",
    "cz",
    "swap",
    "crx",
    "cry",
    "crz",
    "cp",
    "rxx",
    "ryy",
    "rzz"
]

triple_qubit_gates = [
    "ccx",
    "cswap"
]

parametrised_gates = [
    "rx",
    "ry",
    "rz",
    "crx",
    "cry",
    "crz",
    "cp",
    "rxx",
    "ryy",
    "rzz",
]


# Parameters
# -----------------------------------------------------
qubits = 3
initial_circuit_depth = 10
population = 20

runtime_hours = 0
runtime_minutes = 20
time_limit_seconds = (runtime_hours*360 + runtime_minutes*60)/2

# Main
# -----------------------------------------------------
def main():
    # Ensure the experiment results folder exists
    results_folder = "Experiment Results\Logs"
    os.makedirs(results_folder, exist_ok=True)

    # Create a timestamped filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    log_filename = f"Timed_Execution_{time_limit_seconds}s_{timestamp}.log"
    log_filepath = os.path.join(results_folder, log_filename)

    random_max_fitnesses = []
    random_avg_fitnesses = []
    ea_max_fitnesses = []
    ea_avg_fitnesses = []

    # Generate initial chromosomes for EA
    chromosomes = initialize_chromosomes(
        population,
        qubits,
        initial_circuit_depth,
    )

    with open(log_filepath, "w") as log_file:
        log_file.write("Elapsed Time (s),EA Max Fitness,EA Average Fitness,Random Max Fitness,Random Average Fitness\n")

        # RANDOM GENERATOR PHASE
        random_start_time = time.time()
        last_sample_time = 0
        overall_random_max_fitness = 0

        while True:
            elapsed_time_random = time.time() - random_start_time
            if elapsed_time_random > time_limit_seconds:
                break

            # Evaluate random circuits for this iteration
            random_max_fitness, random_avg_fitness, max_fitness_circuit_iter = evaluate_random_circuits(population, 1, qubits, initial_circuit_depth)

            if random_max_fitness > overall_random_max_fitness:
                overall_random_max_fitness = random_max_fitness

            # Log and store fitness only at 10-second intervals
            if elapsed_time_random - last_sample_time >= 10:
                last_sample_time = elapsed_time_random
                random_max_fitnesses.append(overall_random_max_fitness)
                random_avg_fitnesses.append(random_avg_fitness)
                log_file.write(f"{elapsed_time_random:.2f},-,-,{random_max_fitness:.6f},{random_avg_fitness:.6f}\n")

        # EA PHASE
        ea_start_time = time.time()
        last_sample_time = 0
        overall_max_fitness = 0

        while True:
            elapsed_time_ea = time.time() - ea_start_time
            if elapsed_time_ea > time_limit_seconds:
                break

            # Evaluate EA circuits for this iteration
            circuits = get_circuits(chromosomes)
            fitnesses = get_circuit_fitnesses(circuits, qubits)

            max_fitness = max(fitnesses)
            if max_fitness > overall_max_fitness:
                overall_max_fitness = max_fitness
                max_fitness_circuit = circuits[min(range(len(fitnesses)), key=fitnesses.__getitem__)]

            avg_fitness = sum(fitnesses) / len(fitnesses)

            # Log and store fitness only at 10-second intervals
            if elapsed_time_ea - last_sample_time >= 10:
                last_sample_time = elapsed_time_ea
                ea_max_fitnesses.append(max_fitness)
                ea_avg_fitnesses.append(avg_fitness)
                log_file.write(f"{elapsed_time_ea:.2f},{max_fitness:.6f},{avg_fitness:.6f},-,-\n")

            chromosomes = apply_genetic_operators(chromosomes, fitnesses)

        # Plot the results
        plot_results(ea_max_fitnesses, ea_avg_fitnesses, random_max_fitnesses, random_avg_fitnesses)
        print(max_fitness_circuit)

def initialize_chromosomes(population, qubits, initial_circuit_depth):
    chromosomes = []

    for i in range(population):
        chromosome = []

        for i in range(initial_circuit_depth):
            layer = create_new_layer(qubits)
            chromosome.append(layer)

        chromosomes.append(chromosome)

    return chromosomes

def evaluate_random_circuits(population, iterations, qubits, initial_circuit_depth):
    random_circuits = []

    # Generate population * iterations random circuits
    for _ in range(population * iterations):
        chromosome = []
        for _ in range(initial_circuit_depth):
            chromosome.append(create_new_layer(qubits))
        random_circuits.append(chromosome)

    # Convert chromosomes to circuits
    circuits = get_circuits(random_circuits)

    # Evaluate fitness for all circuits
    fitnesses = get_circuit_fitnesses(circuits, qubits)

    # Compute maximum and average fitness
    max_fitness = max(fitnesses)
    avg_fitness = sum(fitnesses) / len(fitnesses)
    max_fitness_circuit = circuits[fitnesses.index(max_fitness)]  # Retrieve the circuit with max fitness

    return max_fitness, avg_fitness, max_fitness_circuit


# Representaion
# -----------------------------------------------------
def get_circuits(circuit_chromosomes):
    circuits = []
    
    for circuit_chromosome in circuit_chromosomes:
        # Initialise circuit
        circuit = QuantumCircuit(len(circuit_chromosome[0]))

        # Gate map for Qiskit Aer native gates with explanations
        chromosome_qiskit_gate_map = {
            "w": lambda qubit: circuit.barrier(qubit),  # Barrier (used for blank "wires")
            "-": None,  # Placeholder for control qubits (no operation)
            "x": lambda qubit: circuit.x(qubit),  # Pauli-X (NOT) gate
            "y": lambda qubit: circuit.y(qubit),  # Pauli-Y gate
            "z": lambda qubit: circuit.z(qubit),  # Pauli-Z gate
            "h": lambda qubit: circuit.h(qubit),  # Hadamard gate
            "s": lambda qubit: circuit.s(qubit),  # S (Phase) gate: R_z(π/2)
            "sdg": lambda qubit: circuit.sdg(qubit),  # S-dagger (Inverse Phase) gate: R_z(-π/2)
            "t": lambda qubit: circuit.t(qubit),  # T gate: R_z(π/4)
            "tdg": lambda qubit: circuit.tdg(qubit),  # T-dagger gate: R_z(-π/4)
            "rx": lambda qubit, theta: circuit.rx(theta, qubit),  # Rotation around the X axis: R_x(θ)
            "ry": lambda qubit, theta: circuit.ry(theta, qubit),  # Rotation around the Y axis: R_y(θ)
            "rz": lambda qubit, theta: circuit.rz(theta, qubit),  # Rotation around the Z axis: R_z(θ)
            "cx": lambda control_qubit, target_qubit: circuit.cx(control_qubit, target_qubit),  # CNOT (Controlled-X) gate
            "cy": lambda control_qubit, target_qubit: circuit.cy(control_qubit, target_qubit),  # Controlled-Y gate
            "cz": lambda control_qubit, target_qubit: circuit.cz(control_qubit, target_qubit),  # Controlled-Z gate
            "swap": lambda q1, q2: circuit.swap(q1, q2),  # SWAP gate (exchange qubits)
            "ccx": lambda q1, q2, target_qubit: circuit.ccx(q1, q2, target_qubit),  # Toffoli gate (Controlled-Controlled-X)
            "cswap": lambda control_qubit, q1, q2: circuit.cswap(control_qubit, q1, q2),  # Controlled-SWAP gate
            "crx": lambda control_qubit, target_qubit, theta: circuit.crx(theta, control_qubit, target_qubit),  # Controlled-RX rotation gate
            "cry": lambda control_qubit, target_qubit, theta: circuit.cry(theta, control_qubit, target_qubit),  # Controlled-RY rotation gate
            "crz": lambda control_qubit, target_qubit, theta: circuit.crz(theta, control_qubit, target_qubit),  # Controlled-RZ rotation gate
            "cp": lambda control_qubit, target_qubit, theta: circuit.cp(theta, control_qubit, target_qubit),  # Controlled-Phase gate
            "rxx": lambda q1, q2, theta: circuit.rxx(theta, q1, q2),  # Ising interaction: R_xx(θ) (rotation on the XX interaction)
            "ryy": lambda q1, q2, theta: circuit.ryy(theta, q1, q2),  # Ising interaction: R_yy(θ) (rotation on the YY interaction)
            "rzz": lambda q1, q2, theta: circuit.rzz(theta, q1, q2),  # Ising interaction: R_zz(θ) (rotation on the ZZ interaction)
        }

        # Helper to apply gates
        for block in circuit_chromosome:
            for qubit in range(len(block)):
                gate_spec = block[qubit]
                if gate_spec == "-":
                    continue
                elif "(" in gate_spec:
                    gate, args = re.match(r"(\w+)\((.+)\)", gate_spec).groups()
                    if gate in parametrised_gates:
                        args = list(args.split(","))
                        args[-1] = float(args[-1])
                        args[:-1] = map(int, args[:-1])
                    else:
                        args = list(map(int, args.split(",")))
                    chromosome_qiskit_gate_map[gate](*args)
                else:
                    chromosome_qiskit_gate_map[gate_spec](qubit)
        circuit.save_statevector()

        circuits.append(circuit.copy())

    return circuits


# Fitness Function
# -----------------------------------------------------
def get_circuit_fitnesses(circuits, qubits):
    """Evaluate fitness of circuits based on similarity to QFT output phases."""
    fitnesses = []

    # Simulate the target QFT circuit and get output states
    target_states = get_qft_target_states(qubits)

    # Simulate each candidate circuit and compare with the target states
    simulator = AerSimulator(method='statevector')
    initial_states = [Statevector.from_label(f"{i:0{qubits}b}") for i in range(2**qubits)]

    for circuit in circuits:
        circuit_states = []
        for state in initial_states:
            new_circuit = QuantumCircuit(*circuit.qregs)
            new_circuit.initialize(state)
            new_circuit.compose(circuit, inplace=True)
            result = simulator.run(new_circuit).result()
            output_state = result.get_statevector()
            circuit_states.append(output_state)

        # Compute fitness based on phase differences
        fitness = compute_phase_fitness(circuit_states, target_states)
        fitnesses.append(fitness)

    return fitnesses

def get_qft_target_states(qubits):
    """Simulate QFT for all computational basis states."""
    simulator = AerSimulator(method='statevector')
    target_states = []

    for i in range(2**qubits):
        state_binary = f"{i:0{qubits}b}"
        target_circuit = QuantumCircuit(qubits)
        for j, bit in enumerate(state_binary):
            if bit == "1":
                target_circuit.x(j)
        target_circuit.append(QFT(num_qubits=qubits), range(qubits))
        target_circuit = transpile(target_circuit, basis_gates=['u', 'cx'])
        target_circuit.save_statevector()
        result = simulator.run(target_circuit).result()
        target_states.append(result.get_statevector())
    return target_states

def compute_phase_fitness(circuit_states, target_states):
    """Computes fitness using state fidelity and phase differences."""
    fitness = 0
    for circuit_state, target_state in zip(circuit_states, target_states):
        #fidelity = state_fidelity(circuit_state, target_state)
        fidelity = phase_sensitive_fidelity(circuit_state, target_state)
        fitness += fidelity  # Combine fitness based on fidelity
    fitness /= len(target_states)  # Normalise fitness to [0, 1]
    return fitness

def phase_sensitive_fidelity(output_state, target_state):
    # Ensure input states are Qiskit Statevectors
    output_sv = Statevector(output_state)
    target_sv = Statevector(target_state)
    
    # Get the computational basis states and their amplitudes
    output_amplitudes = output_sv.data
    target_amplitudes = target_sv.data
    
    # Extract phases, ignoring amplitudes
    output_phases = np.angle(output_amplitudes)
    target_phases = np.angle(target_amplitudes)
    
    # Compute phase differences (modulo 2π to handle wrapping)
    phase_differences = (output_phases - target_phases) % (2 * np.pi)
    
    # Map phase differences to the range [-π, π] for meaningful comparison
    phase_differences = np.where(phase_differences > np.pi, 
                                 phase_differences - 2 * np.pi, 
                                 phase_differences)
    
    # Compute a fidelity metric based on the phase differences
    # Example: Mean squared phase error
    phase_fidelity = 1 - np.mean(phase_differences**2) / (np.pi**2)
    
    return phase_fidelity


# Genetic Operators
# -----------------------------------------------------
def apply_genetic_operators(chromosomes, fitnesses, parent_chromosomes=population//4):  
    # Sort chromosomes by fitness in descending order
    sorted_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)

    # Preserve the top parent_chromosomes chromosomes (deep copy to avoid mutation)
    elites = [copy.deepcopy(chromosomes[idx]) for idx in sorted_indices[:parent_chromosomes]]

    # Get indices for crossover and mutation
    top_indices = sorted_indices[:parent_chromosomes]
    bottom_indices = sorted_indices[-(len(chromosomes) - parent_chromosomes):]

    # Generate children through crossover and mutation
    child_chromosomes = []
    while len(child_chromosomes) < len(bottom_indices):
        parent_1_index, parent_2_index = np.random.choice(top_indices, 2, replace=False)
        child_1, child_2 = crossover(chromosomes[parent_1_index], chromosomes[parent_2_index])
        child_chromosomes.append(mutate_chromosome(child_1))
        if len(child_chromosomes) < len(bottom_indices):
            child_chromosomes.append(mutate_chromosome(child_2))

    # Replace bottom chromosomes with children and append elites
    new_population = elites + child_chromosomes

    return new_population

def crossover(parent_1, parent_2):
    """Single-point crossover between two chromosomes."""
    crossover_point = np.random.randint(1, len(parent_1))
    child_1 = parent_1[:crossover_point] + parent_2[crossover_point:]
    child_2 = parent_2[:crossover_point] + parent_1[crossover_point:]
    return child_1, child_2

def mutate_chromosome(
    chromosome,
    parameter_mutation_rate=0.1,
    gate_mutation_rate=0.1,
    layer_mutation_rate=0.1,
    max_parameter_mutation=0.1,
    layer_deletion_rate=0.05  # Probability of deleting a layer
):
    """Mutates a single chromosome with a given mutation rate, including gate removal and layer deletion."""

    for i, layer in enumerate(chromosome):
        # Chance to delete the entire layer
        if np.random.rand() < layer_deletion_rate:
            chromosome[i] = ["w"] * len(layer)  # Replace with a blank layer (barriers only)
            continue

        for j in range(len(layer)):
            # Chance to mutate the gate type
            if np.random.rand() < gate_mutation_rate:
                gate_type = np.random.choice(["single", "double", "triple", "remove"])
                if gate_type == "remove":
                    layer[j] = "w"  # Replace with a barrier to represent gate removal
                elif gate_type == "single":
                    gate_choice = np.random.choice(single_qubit_gates)
                    if gate_choice in parametrised_gates:
                        layer[j] = gate_choice + f"({j}, {np.random.random()})"
                    else:
                        layer[j] = gate_choice + f"({j})"
                elif gate_type == "double":
                    target_qubit = np.random.randint(0, len(layer))
                    control_qubit = np.random.randint(0, len(layer))
                    while control_qubit == target_qubit:
                        control_qubit = np.random.randint(0, len(layer))
                    gate_choice = np.random.choice(double_qubit_gates)
                    if gate_choice in parametrised_gates:
                        layer[target_qubit] = gate_choice + f"({control_qubit},{target_qubit},{np.random.random()})"
                    else:
                        layer[target_qubit] = gate_choice + f"({control_qubit},{target_qubit})"
                    layer[control_qubit] = "-"
                elif gate_type == "triple" and len(layer) >= 3:
                    qubit_1, qubit_2, qubit_3 = np.random.choice(len(layer), size=3, replace=False)
                    gate_choice = np.random.choice(triple_qubit_gates)
                    if gate_choice in parametrised_gates:
                        layer[qubit_3] = gate_choice + f"({qubit_1},{qubit_2},{qubit_3},{np.random.random()})"
                    else:
                        layer[qubit_3] = gate_choice + f"({qubit_1},{qubit_2},{qubit_3})"
                    layer[qubit_1] = "-"
                    layer[qubit_2] = "-"

            # Chance to mutate gate parameters
            elif np.random.rand() < parameter_mutation_rate:
                match = re.match(r"([a-z]+)\((.*)\)", layer[j])
                if match and match.group(1) in parametrised_gates:
                    gate_name, params = match.groups()
                    params_list = params.split(",")
                    param_value = float(params_list[-1])

                    # Apply a random factor to mutate the parameter
                    factor = np.random.uniform(1, max_parameter_mutation)
                    if np.random.rand() < 0.5:
                        param_value *= factor
                    else:
                        param_value /= factor
                    param_value = max(param_value, 0.0)

                    params_list[-1] = str(param_value)
                    layer[j] = f"{gate_name}({','.join(params_list)})"

    # Chance to add a new layer
    if np.random.rand() < layer_mutation_rate:
        new_layer = create_new_layer(len(chromosome[0]))
        chromosome.append(new_layer)

    return chromosome

def create_new_layer(qubits):
    layer = ["w"] * qubits  # Initialize the layer with "w"

    # Apply single-qubit gates
    for qubit in range(qubits):
        if np.random.rand() < 0.7:  # High probability for "w"
            layer[qubit] = "w"
        else:
            gate_choice = np.random.choice(single_qubit_gates)
            if gate_choice in parametrised_gates:
                layer[qubit] = gate_choice + f"({qubit}, {np.random.random()})"
            else:
                layer[qubit] = gate_choice + f"({qubit})"

    # Apply double-qubit gates
    if np.random.rand() < 0.5:  # Moderate probability for double-qubit gates
        target_qubit = np.random.randint(0, qubits)
        control_qubit = np.random.randint(0, qubits)

        while control_qubit == target_qubit:  # Ensure control and target are different
            control_qubit = np.random.randint(0, qubits)

        gate_choice = np.random.choice(double_qubit_gates)
        if gate_choice in parametrised_gates:
            layer[target_qubit] = gate_choice + f"({control_qubit},{target_qubit},{np.random.random()})"
        else:
            layer[target_qubit] = gate_choice + f"({control_qubit},{target_qubit})"
        layer[control_qubit] = "-"  # Mark control qubit

    # Apply triple-qubit gates
    if np.random.rand() < 0.2:  # Lower probability for triple-qubit gates
        qubit_1, qubit_2, qubit_3 = np.random.choice(range(qubits), size=3, replace=False)
        gate_choice = np.random.choice(triple_qubit_gates)
        if gate_choice in parametrised_gates:
            layer[qubit_3] = gate_choice + f"({qubit_1},{qubit_2},{qubit_3},{np.random.random()})"
        else:
            layer[qubit_3] = gate_choice + f"({qubit_1},{qubit_2},{qubit_3})"
        layer[qubit_1] = "-"
        layer[qubit_2] = "-"

    return layer


# Plotting Function
# -----------------------------------------------------
def plot_results(ea_max_fitnesses, ea_avg_fitnesses, random_max_fitnesses, random_avg_fitnesses):
    # Ensure the overall random max fitness is consistent
    overall_random_max_fitness = max(random_max_fitnesses, default=0)

    plt.figure(figsize=(10, 6))
    elapsed_intervals = np.arange(0, len(ea_max_fitnesses) * 10, 10)

    # Plot EA Fitnesses
    plt.plot(elapsed_intervals, ea_max_fitnesses, label='EA Max Fitness', color='blue', marker='o')
    plt.plot(elapsed_intervals, ea_avg_fitnesses, label='EA Avg Fitness', color='cyan', linestyle='--')

    # Plot Random Fitnesses
    random_intervals = np.arange(0, len(random_max_fitnesses) * 10, 10)
    plt.plot(random_intervals, random_max_fitnesses, label='Random Max Fitness', color='red', marker='x')
    plt.plot(random_intervals, random_avg_fitnesses, label='Random Avg Fitness', color='orange', linestyle='--')

    # Set axis limits
    plt.ylim(0, 1)
    plt.xlim(0, max(max(elapsed_intervals, default=0), max(random_intervals, default=0)) + 10)

    # Label axes and add legend
    plt.xlabel("Elapsed Time (s)")
    plt.ylabel("Fitness")
    plt.title("Fitness Progress Over Time")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Create a timestamped filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    plt.savefig("Experiment Results\Charts\Timed_Execution_" + timestamp + ".png")


if __name__ == "__main__":  
    main()