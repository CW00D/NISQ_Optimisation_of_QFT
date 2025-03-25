"""
This module implements an evolutionary algorithm (EA) approach to optimize quantum circuits.
It includes functions for initializing a population of chromosomes (each representing a circuit),
evaluating circuit fitness (using state fidelity against QFT-transformed states), and applying
genetic operators (elitism, crossover, and mutation).

The code is organized into:
    - Global constants (gate lists and simulator)
    - Helper functions for parsing gate specifications and building gate maps
    - Initialization and evaluation functions
    - Circuit representation functions
    - Fitness functions
    - Genetic operator functions
"""

# ---------------------------
# Standard Library Imports
# ---------------------------
import re
import random

# ---------------------------
# Third-Party Imports
# ---------------------------
import numpy as np

# ---------------------------
# Qiskit Imports
# ---------------------------
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QFT
from qiskit.quantum_info import state_fidelity

# ---------------------------
# Custom noise model
# ---------------------------
from qiskit_aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error, phase_damping_error

# Define realistic error rates for single-qubit gates (in probability)
# Note: RZ is often implemented virtually, so we assume zero error.
# Define realistic error rates for single-qubit gates
single_qubit_error_rates = {
    "x": 0.002, "y": 0.002, "z": 0.001, "h": 0.0015, "s": 0.001, "sdg": 0.001,
    "t": 0.0015, "tdg": 0.0015, "rx": 0.002, "ry": 0.002, "rz": 0.0  # RZ assumed error-free
}

# Define realistic error rates for two-qubit gates
two_qubit_error_rates = {
    "cx": 0.02, "cy": 0.02, "cz": 0.018, "swap": 0.025,
    "crx": 0.02, "cry": 0.02, "crz": 0.02, "cp": 0.02,
    "rxx": 0.02, "ryy": 0.02, "rzz": 0.02
}

# Define realistic error rates for three-qubit gates
three_qubit_error_rates = {
    "ccx": 0.1,    # Estimated ~6× CX error
    "cswap": 0.12  # Estimated ~8× CX error
}

# Define amplitude and phase damping parameters for single-qubit gates.
# These represent the probability of relaxation or dephasing during the gate.
amp_gamma = 0.002    # 0.2% amplitude damping error
phase_gamma = 0.002  # 0.2% phase damping error

# Create a new noise model instance
noise_model = NoiseModel()

# Apply single-qubit errors (combine depolarizing, amplitude damping, and phase damping)
for gate, rate in single_qubit_error_rates.items():
    if rate > 0:
        # Create individual noise channels
        error_depol = depolarizing_error(rate, 1)
        error_amp = amplitude_damping_error(amp_gamma)
        error_phase = phase_damping_error(phase_gamma)
        # Combine the errors sequentially (order can be tuned)
        combined_error = error_depol.compose(error_amp).compose(error_phase)
        # Add the noise model for this gate
        noise_model.add_all_qubit_quantum_error(combined_error, gate)
    else:
        # If the error rate is zero (e.g., for RZ), we skip adding noise
        continue

# Apply two-qubit errors using only depolarizing noise
for gate, rate in two_qubit_error_rates.items():
    error_2q = depolarizing_error(rate, 2)
    noise_model.add_all_qubit_quantum_error(error_2q, gate)


# ---------------------------
# Global Constants
# ---------------------------
POPULATION = 100
ELITISM_NUMBER = POPULATION // 3
INITIAL_CIRCUIT_DEPTH = 10
PARAMETER_MUTATION_RATE = 0.1
GATE_MUTATION_RATE = 0.3
LAYER_MUTATION_RATE = 0.2
MAX_PARAMETER_MUTATION = 0.2
LAYER_DELETION_RATE = 0.03

SINGLE_QUBIT_GATES = [
    "x", "y", "z", "h", "s", "sdg", "t", "tdg", "rx", "ry", "rz"
]
DOUBLE_QUBIT_GATES = [
    "cx", "cy", "cz", "swap", "crx", "cry", "crz", "cp", "rxx", "ryy", "rzz"
]
TRIPLE_QUBIT_GATES = [
    "ccx", "cswap"
]
PARAMETRISED_GATES = [
    "rx", "ry", "rz", "crx", "cry", "crz", "cp", "rxx", "ryy", "rzz"
]

# Qiskit Simulator instance.
NOISY_SIMULATOR = AerSimulator(noise_model=noise_model, method='statevector')
NOISLESS_SIMULATOR = AerSimulator(method='statevector')

native_gates = NOISLESS_SIMULATOR.configuration().basis_gates

# Fitness Cache
fitness_cache = {}

# ---------------------------
# Helper Functions
# ---------------------------
def parse_gate_spec(gate_spec):
    """
    Parse a gate specification string.

    Parameters:
        gate_spec (str): A string representing the gate (e.g. "rx(0, 0.5)").

    Returns:
        tuple: (gate_name, args) where args is a list of arguments (or None if no arguments).
               For parametrised gates, all but the last argument are converted to int and the last to float.
    """
    if "(" in gate_spec:
        match = re.match(r"(\w+)\((.+)\)", gate_spec)
        if not match:
            raise ValueError(f"Invalid gate specification: {gate_spec}")
        gate, args_str = match.groups()
        args_list = [arg.strip() for arg in args_str.split(",")]
        if gate in PARAMETRISED_GATES:
            # Convert all but the last argument to int, last to float.
            converted_args = list(map(int, args_list[:-1])) + [float(args_list[-1])]
        else:
            converted_args = list(map(int, args_list))
        return gate, converted_args
    else:
        return gate_spec, None

def build_gate_map(circuit):
    """
    Build a mapping from gate names to lambda functions for applying them to the given circuit.

    Parameters:
        circuit (QuantumCircuit): The circuit to which the gates will be applied.

    Returns:
        dict: Mapping of gate name to a function.
    """
    return {
        "w": lambda qubit: circuit.barrier(qubit),  # "w" is a barrier (placeholder)
        "-": lambda qubit: None,  # no operation for control markers
        "x": lambda qubit: circuit.x(qubit),
        "y": lambda qubit: circuit.y(qubit),
        "z": lambda qubit: circuit.z(qubit),
        "h": lambda qubit: circuit.h(qubit),
        "s": lambda qubit: circuit.s(qubit),
        "sdg": lambda qubit: circuit.sdg(qubit),
        "t": lambda qubit: circuit.t(qubit),
        "tdg": lambda qubit: circuit.tdg(qubit),
        "rx": lambda qubit, theta: circuit.rx(theta, qubit),
        "ry": lambda qubit, theta: circuit.ry(theta, qubit),
        "rz": lambda qubit, theta: circuit.rz(theta, qubit),
        "cx": lambda control, target: circuit.cx(control, target),
        "cy": lambda control, target: circuit.cy(control, target),
        "cz": lambda control, target: circuit.cz(control, target),
        "swap": lambda q1, q2: circuit.swap(q1, q2),
        "ccx": lambda q1, q2, target: circuit.ccx(q1, q2, target),
        "cswap": lambda control, q1, q2: circuit.cswap(control, q1, q2),
        "crx": lambda control, target, theta: circuit.crx(theta, control, target),
        "cry": lambda control, target, theta: circuit.cry(theta, control, target),
        "crz": lambda control, target, theta: circuit.crz(theta, control, target),
        "cp": lambda control, target, theta: circuit.cp(theta, control, target),
        "rxx": lambda q1, q2, theta: circuit.rxx(theta, q1, q2),
        "ryy": lambda q1, q2, theta: circuit.ryy(theta, q1, q2),
        "rzz": lambda q1, q2, theta: circuit.rzz(theta, q1, q2),
    }

def rank_selection(chromosomes, fitnesses):
    """
    Selects one chromosome using linear rank selection.
    The highest fitness chromosome has rank 'pop_size',
    the lowest has rank 1, and selection probability is proportional to rank.
    """
    pop_size = len(chromosomes)
    # Sort indices by fitness ascending
    sorted_indices = sorted(range(pop_size), key=lambda i: fitnesses[i])
    
    # ranks[i] = the rank of the chromosome sorted_indices[i]
    # highest fitness -> rank = pop_size, lowest fitness -> rank = 1
    ranks = [i+1 for i in range(pop_size)]  # ranks from 1..pop_size
    
    # Build a list of (chromosome, rank)
    # sorted_indices[0] is the worst, has rank=1
    # sorted_indices[-1] is the best, has rank=pop_size
    # We want the best to have the highest rank
    chrom_and_rank = []
    for rank_idx, idx in enumerate(sorted_indices):
        chrom_and_rank.append((idx, ranks[rank_idx]))
    
    total_rank = sum(ranks)
    pick = random.uniform(0, total_rank)
    current = 0
    
    for idx, rank_val in chrom_and_rank:
        current += rank_val
        if current >= pick:
            return copy_chromosome(chromosomes[idx])    
    # Fallback (rarely triggered due to float rounding):
    return copy_chromosome(chromosomes[sorted_indices[-1]])

def get_chromosome_key(chromosome):
    """
    Convert a chromosome (list of layers) into a hashable tuple-of-tuples.
    """
    return tuple(tuple(layer) for layer in chromosome)

def copy_chromosome(chromosome):
    # Slicing each inner list creates a new list while the strings (immutable) remain shared.
    return [layer[:] for layer in chromosome]

# ---------------------------
# Initialization and Evaluation Functions
# ---------------------------
def initialize_chromosomes(qubits):
    """
    Initialize a population of chromosomes.

    Each chromosome is a list of layers, where each layer is a list of gate specification strings.

    Parameters:
        qubits (int): Number of qubits (i.e. gates per layer).
        INITIAL_CIRCUIT_DEPTH (int): Number of layers per chromosome.

    Returns:
        list: List of chromosomes.
    """
    return [[create_new_layer(qubits) for _ in range(INITIAL_CIRCUIT_DEPTH)]
            for _ in range(POPULATION)]

def evaluate_random_circuits(iterations, qubits, target_states):
    """
    Evaluate a set of randomly generated circuits.

    Parameters:
        iterations (int): Number of iterations.
        qubits (int): Number of qubits.
        initial_states (list): List of initial state Statevectors.
        target_states (list): List of target state Statevectors (from QFT).

    Returns:
        tuple: (max_fitness, average_fitness) over the evaluated circuits.
    """
    random_chromosomes = [[create_new_layer(qubits) for _ in range(INITIAL_CIRCUIT_DEPTH)]
                       for _ in range(POPULATION * iterations)]
    random_circuits = get_circuits(random_chromosomes)
    fitnesses = get_circuit_fitnesses(target_states, random_circuits, random_chromosomes)
    return max(fitnesses), sum(fitnesses) / len(fitnesses)

def get_qft_target_states(qubits, simulator=NOISLESS_SIMULATOR):
    """
    Simulate the QFT for all computational basis states and return their statevectors.

    Parameters:
        qubits (int): Number of qubits.
        simulator (AerSimulator): The Qiskit simulator.

    Returns:
        list: List of statevectors corresponding to QFT outputs.
    """
    target_states = []
    for i in range(2 ** qubits):
        state_binary = f"{i:0{qubits}b}"
        target_circuit = QuantumCircuit(qubits)
        for j, bit in enumerate(state_binary):
            if bit == "1":
                target_circuit.x(j)
        target_circuit.append(QFT(num_qubits=qubits), list(range(qubits)))
        target_circuit = transpile(target_circuit, basis_gates=['u', 'cx'])
        target_circuit.save_density_matrix()
        result = simulator.run(target_circuit).result()
        target_states.append(result.data(0)['density_matrix'])
    return target_states


# ---------------------------
# Circuit Representation
# ---------------------------
def get_circuits(chromosome_list):
    """
    Convert a list of chromosomes (each a list of layers) into Qiskit QuantumCircuit objects.

    Parameters:
        chromosome_list (list): List of chromosomes.

    Returns:
        list: List of QuantumCircuit objects.
    """
    circuits = []
    for chromosome in chromosome_list:
        num_qubits = len(chromosome[0])
        circuit = QuantumCircuit(num_qubits)
        gate_map = build_gate_map(circuit)
        for layer in chromosome:
            for qubit, gate_spec in enumerate(layer):
                if gate_spec == "-":
                    continue
                gate, args = parse_gate_spec(gate_spec)
                if args is not None:
                    gate_map[gate](*args)
                else:
                    gate_map[gate](qubit)
        circuits.append(circuit.copy())
    return circuits


# ---------------------------
# Fitness Functions
# ---------------------------
def get_circuit_fitnesses(target_states, circuits, chromosomes, simulator=NOISY_SIMULATOR, depth_lambda=0.005):
    """
    Evaluate the fitness for each circuit (corresponding to the provided chromosomes).
    For chromosomes that have been evaluated before, use the cached fitness.
    
    Parameters:
        target_states (list): List of target density matrices.
        circuits (list): List of QuantumCircuit objects.
        chromosomes (list): List of chromosome representations corresponding to the circuits.
        simulator (AerSimulator): The Qiskit simulator.
    
    Returns:
        list: Fitness values for each circuit.
    """
    fitnesses = [None] * len(circuits)
    batch_circuits = []
    batch_indices = []
    # Use all computational basis states; group_size is 2^(number of qubits)
    group_size = 2 ** len(circuits[0].qregs[0])
    
    for i, circuit in enumerate(circuits):
        key = get_chromosome_key(chromosomes[i])
        if key in fitness_cache:
            fitnesses[i] = fitness_cache[key]
        else:
            batch_indices.append(i)
            # Transpile the circuit once for this chromosome
            transpiled_circuit = transpile(circuit, basis_gates=native_gates)
            # For each computational basis state, prepare that state using X gates.
            for idx in range(group_size):
                new_circuit = QuantumCircuit(*circuit.qregs)
                binary_str = format(idx, f'0{len(circuit.qregs[0])}b')
                for j, bit in enumerate(binary_str):
                    if bit == '1':
                        new_circuit.x(j)
                new_circuit.compose(transpiled_circuit, inplace=True)
                new_circuit.save_density_matrix()
                batch_circuits.append(new_circuit)
    
    if batch_circuits:
        results = simulator.run(batch_circuits).result()
        output_states = [results.data(j)['density_matrix'] for j in range(len(batch_circuits))]
        for count, i in enumerate(batch_indices):
            start = count * group_size
            end = start + group_size
            circuit_states = output_states[start:end]
            fidelity = compute_fidelity(circuit_states, target_states)
            depth_penalty = depth_lambda * len(chromosomes[i])
            fitness = fidelity - depth_penalty
            key = get_chromosome_key(chromosomes[i])
            fitness_cache[key] = fitness
            fitnesses[i] = fitness
    return fitnesses

def compute_fidelity(circuit_states, target_states):
    """
    Compute the fitness of a circuit as the average state fidelity over all basis states.

    Parameters:
        circuit_states (list): List of output statevectors for a circuit.
        target_states (list): List of target statevectors.

    Returns:
        float: Fitness value between 0 and 1.
    """
    fitness = 0
    for out_state, target_state in zip(circuit_states, target_states):
        fitness += state_fidelity(out_state, target_state)
    return fitness / len(target_states)


# ---------------------------
# Genetic Operators
# ---------------------------
def apply_genetic_operators(chromosomes, fitnesses):
    """
    Creates a new population by preserving 'ELITISM_NUMBER' top individuals,
    then using rank selection to fill the rest.
    """
    # Identify elites
    sorted_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)
    elites = [copy_chromosome(chromosomes[i]) for i in sorted_indices[:ELITISM_NUMBER]]
    
    # Generate the remaining population with rank selection
    new_population = []
    remaining = len(chromosomes) - ELITISM_NUMBER
    while len(new_population) < remaining:
        parent1 = rank_selection(chromosomes, fitnesses)
        parent2 = rank_selection(chromosomes, fitnesses)
        child1, child2 = crossover(parent1, parent2)
        
        child1 = mutate_chromosome(child1)
        child2 = mutate_chromosome(child2)
        
        new_population.append(child1)
        if len(new_population) < remaining:
            new_population.append(child2)
    
    return elites + new_population

def crossover(parent_1, parent_2):
    """
    Perform single-point crossover between two chromosomes.

    Parameters:
        parent_1 (list): First parent's chromosome.
        parent_2 (list): Second parent's chromosome.

    Returns:
        tuple: Two child chromosomes.
    """
    if len(parent_1) > 1:
        crossover_point = np.random.randint(1, len(parent_1))
        child_1 = parent_1[:crossover_point] + parent_2[crossover_point:]
        child_2 = parent_2[:crossover_point] + parent_1[crossover_point:]
    else:
        child_1, child_2 = parent_1, parent_2
    return child_1, child_2

def mutate_chromosome(chromosome):
    """
    Mutate a single chromosome by modifying gate types, parameters, or layers.
    
    If a layer is selected for deletion, it is removed from the chromosome.
    
    Parameters:
        chromosome (list): The chromosome to mutate.
    
    Returns:
        list: Mutated chromosome.
    """
    mutated_chromosome = []
    
    for layer in chromosome:
        # If the deletion condition is met, skip this layer entirely.
        if np.random.rand() < LAYER_DELETION_RATE:
            continue
        else:
            # Process gate mutations within the layer.
            for j in range(len(layer)):
                # Skip if it's a control qubit.
                if layer[j] == "-":
                    continue
                elif np.random.rand() < GATE_MUTATION_RATE:
                    gate_type = np.random.choice(["single", "double", "triple", "remove"])
                    if gate_type == "remove":
                        layer[j] = "w"
                    elif gate_type == "single":
                        gate_choice = np.random.choice(SINGLE_QUBIT_GATES)
                        if gate_choice in PARAMETRISED_GATES:
                            layer[j] = f"{gate_choice}({j}, {np.random.random()})"
                        else:
                            layer[j] = f"{gate_choice}({j})"
                    elif gate_type == "double":
                        target_qubit = np.random.randint(0, len(layer))
                        control_qubit = np.random.randint(0, len(layer))
                        while control_qubit == target_qubit:
                            control_qubit = np.random.randint(0, len(layer))
                        gate_choice = np.random.choice(DOUBLE_QUBIT_GATES)
                        if gate_choice in PARAMETRISED_GATES:
                            layer[target_qubit] = f"{gate_choice}({control_qubit},{target_qubit},{np.random.random()})"
                        else:
                            layer[target_qubit] = f"{gate_choice}({control_qubit},{target_qubit})"
                        layer[control_qubit] = "-"
                    elif gate_type == "triple" and len(layer) >= 3:
                        qubit_1, qubit_2, qubit_3 = np.random.choice(len(layer), size=3, replace=False)
                        gate_choice = np.random.choice(TRIPLE_QUBIT_GATES)
                        if gate_choice in PARAMETRISED_GATES:
                            layer[qubit_3] = f"{gate_choice}({qubit_1},{qubit_2},{qubit_3},{np.random.random()})"
                        else:
                            layer[qubit_3] = f"{gate_choice}({qubit_1},{qubit_2},{qubit_3})"
                        layer[qubit_1] = "-"
                        layer[qubit_2] = "-"
                elif np.random.rand() < PARAMETER_MUTATION_RATE:
                    match = re.match(r"([a-z]+)\((.*)\)", layer[j])
                    if match and match.group(1) in PARAMETRISED_GATES:
                        gate_name, params_str = match.groups()
                        params_list = [p.strip() for p in params_str.split(",")]
                        param_value = float(params_list[-1])
                        factor = np.random.uniform(1, MAX_PARAMETER_MUTATION)
                        if np.random.rand() < 0.5:
                            param_value *= factor
                        else:
                            param_value /= factor
                        param_value = max(param_value, 0.0)
                        params_list[-1] = str(param_value)
                        layer[j] = f"{gate_name}({','.join(params_list)})"
            # Add the (possibly mutated) layer to the new chromosome.
            mutated_chromosome.append(layer)
    
    # Potentially add a new layer.
    if np.random.rand() < LAYER_MUTATION_RATE:
        # Assuming chromosome isn't empty; if it might be, adjust accordingly.
        new_layer = create_new_layer(len(chromosome[0]))
        mutated_chromosome.append(new_layer)
    
    # If the resulting chromosome is empty, replace it with a new randomly generated chromosome.
    if not mutated_chromosome:
        mutated_chromosome = initialize_chromosomes(len(chromosome[0]))[0]
    
    return mutated_chromosome

def create_new_layer(qubits):
    """
    Create a new layer for a chromosome with a single decision per qubit.
    
    For each position, one gate is chosen from a combined list:
      - "w" (barrier) with probability 0.25.
      - All single-qubit, double-qubit, and triple-qubit gates share the remaining 0.75 equally.
    
    For a double-qubit gate, one additional partner is chosen from the remaining free indices.
    For a triple-qubit gate, two additional partners are chosen.
    If there aren't enough free indices, the choice falls back to a barrier.
    
    Returns:
        list: A new layer represented as a list of gate specification strings.
    """
    # Combined list of available gates.
    available_gates = ["w"] + SINGLE_QUBIT_GATES + DOUBLE_QUBIT_GATES + TRIPLE_QUBIT_GATES
    n_options = len(available_gates)
    
    # Set barrier weight to 0.25; the remaining 0.75 is distributed equally among the other options.
    weight_barrier = 0.25
    weight_other = 0.75 / (n_options - 1) if n_options > 1 else 0
    weights = [weight_barrier] + [weight_other] * (n_options - 1)
    
    # Initialize a layer with None for each qubit.
    layer = [None] * qubits
    free_indices = list(range(qubits))
    
    while free_indices:
        idx = free_indices.pop(0)
        
        # Skip if this qubit has already been assigned as a control ("-")
        if layer[idx] is not None:
            continue
        
        chosen_gate = random.choices(available_gates, weights=weights, k=1)[0]
        
        if chosen_gate in DOUBLE_QUBIT_GATES:
            if free_indices:
                partner = random.choice(free_indices)
                free_indices.remove(partner)
                
                if chosen_gate in PARAMETRISED_GATES:
                    layer[idx] = f"{chosen_gate}({partner},{idx},{np.random.random()})"
                else:
                    layer[idx] = f"{chosen_gate}({partner},{idx})"
                
                # Mark the control qubit, ensuring it hasn’t been assigned anything else
                if layer[partner] is None:
                    layer[partner] = "-"
            else:
                layer[idx] = "w"  # Not enough free indices.
        
        elif chosen_gate in TRIPLE_QUBIT_GATES:
            if len(free_indices) >= 2:
                partners = random.sample(free_indices, 2)
                for p in partners:
                    free_indices.remove(p)
                
                if chosen_gate in PARAMETRISED_GATES:
                    layer[idx] = f"{chosen_gate}({partners[0]},{partners[1]},{idx},{np.random.random()})"
                else:
                    layer[idx] = f"{chosen_gate}({partners[0]},{partners[1]},{idx})"
                
                # Mark both partner positions as controls, ensuring they are not overwritten
                if layer[partners[0]] is None:
                    layer[partners[0]] = "-"
                if layer[partners[1]] is None:
                    layer[partners[1]] = "-"
            else:
                layer[idx] = "w"  # Not enough free indices.
        
        else:
            # Ensure we only assign a gate to an unoccupied spot
            if layer[idx] is None:
                if chosen_gate == "w":
                    layer[idx] = "w"
                elif chosen_gate in PARAMETRISED_GATES:
                    layer[idx] = f"{chosen_gate}({idx},{np.random.random()})"
                else:
                    layer[idx] = f"{chosen_gate}({idx})"
    
    return layer
