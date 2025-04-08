"""
This module implements an evolutionary algorithm (EA) approach to optimize quantum circuits.
It includes functions for initializing a population of chromosomes (each representing a circuit),
evaluating circuit fitness (using state fidelity against QFT-transformed states), and applying
genetic operators (elitism, crossover, and mutation). The code is organized into:
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
# Global Constants
# ---------------------------
POPULATION = 100
ELITISM_NUMBER = POPULATION // 3
INITIAL_CIRCUIT_DEPTH = 10
PARAMETER_MUTATION_RATE = 0.1
PARAMETER_STD_DEV = 0.01
GATE_MUTATION_RATE = 0.3
LAYER_MUTATION_RATE = 0.2
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

# Qiskit Simulator instances.
NOISLESS_SIMULATOR = AerSimulator(method='density_matrix')

native_gates = NOISLESS_SIMULATOR.configuration().basis_gates

# Fitness Cache
fitness_cache = {}

# ---------------------------
# Helper Functions
# ---------------------------
def parse_gate_spec(gate_spec):
    """
    Parse a gate specification string.

    For the updated representation:
      - Single-qubit gates are represented as "h" or parameterised as "rx(0.5)".
      - For multi-qubit gates, the control qubit(s) (and any parameters) are included,
        but the target is not provided in the string.

    Parameters:
        gate_spec (str): A string representing the gate.

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
            # For example, "rx(0.5)" or "crx(2,0.5)".
            converted_args = list(map(int, args_list[:-1])) + [float(args_list[-1])]
        else:
            # Non-parameterised multi-qubit gates (e.g., "cx(2)").
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
    sorted_indices = sorted(range(pop_size), key=lambda i: fitnesses[i])
    ranks = [i+1 for i in range(pop_size)]
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
    return copy_chromosome(chromosomes[sorted_indices[-1]])

def get_chromosome_key(chromosome):
    """
    Convert a chromosome (list of layers) into a hashable tuple-of-tuples.
    """
    return tuple(tuple(layer) for layer in chromosome)

def copy_chromosome(chromosome):
    """
    Create a deep copy of a chromosome.
    """
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
    Returns:
        list: List of chromosomes.
    """
    return [[create_new_layer(qubits) for _ in range(INITIAL_CIRCUIT_DEPTH)]
            for _ in range(POPULATION)]

def evaluate_random_circuits(iterations, qubits, target_states):
    """
    Evaluate a set of randomly generated circuits.
    """
    random_chromosomes = [[create_new_layer(qubits) for _ in range(INITIAL_CIRCUIT_DEPTH)]
                            for _ in range(POPULATION * iterations)]
    random_circuits = get_circuits(random_chromosomes)
    fitnesses = get_circuit_fitnesses(target_states, random_circuits, random_chromosomes)
    return max(fitnesses), sum(fitnesses) / len(fitnesses)

def get_qft_target_states(qubits, simulator=NOISLESS_SIMULATOR):
    """
    Simulate the QFT for all computational basis states and return their statevectors.
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
                if gate in DOUBLE_QUBIT_GATES:
                    # In the new representation, args holds only control (and optionally parameter)
                    if args is not None:
                        if len(args) == 1:
                            args.append(qubit)  # Append the target qubit
                        elif len(args) == 2:
                            # For parameterised two-qubit gates, insert the target in the middle.
                            control = args[0]
                            theta = args[1]
                            args = [control, qubit, theta]
                    else:
                        args = [None, qubit]
                    gate_map[gate](*args)
                elif gate in TRIPLE_QUBIT_GATES:
                    # For triple-qubit gates, args holds both controls; append target automatically.
                    if args is not None:
                        if len(args) == 2:
                            args.append(qubit)
                        elif len(args) == 3:
                            args = [args[0], args[1], qubit, args[2]]
                    else:
                        args = [None, None, qubit]
                    gate_map[gate](*args)
                else:
                    # Single-qubit gate (parameterised or not)
                    if args is not None:
                        gate_map[gate](qubit, *args)
                    else:
                        gate_map[gate](qubit)
        circuits.append(circuit.copy())
    return circuits

# ---------------------------
# Fitness Functions
# ---------------------------
def get_circuit_fitnesses(target_states, circuits, chromosomes, simulator=NOISLESS_SIMULATOR, depth_lambda=0.005):
    """
    Evaluate the fitness for each circuit.
    """
    fitnesses = [None] * len(circuits)
    batch_circuits = []
    batch_indices = []
    group_size = 2 ** len(circuits[0].qregs[0])
    
    for i, circuit in enumerate(circuits):
        key = get_chromosome_key(chromosomes[i])
        if key in fitness_cache:
            fitnesses[i] = fitness_cache[key]
        else:
            batch_indices.append(i)
            transpiled_circuit = transpile(circuit, basis_gates=native_gates)
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
    Creates a new population by preserving top individuals (elitism) and using rank selection.
    """
    sorted_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)
    elites = [copy_chromosome(chromosomes[i]) for i in sorted_indices[:ELITISM_NUMBER]]
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
    # Apply fixer function to clean up control markers before returning the new population
    return [fix_control_placeholders(chromo) for chromo in elites + new_population]

def crossover(parent_1, parent_2):
    """
    Perform single-point crossover between two chromosomes.
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
    Mutate a chromosome by modifying gate types, parameters, or layers.
    This version uses the updated representation (without target qubits in the strings)
    and ensures that "-" markers for controls are assigned only when needed.
    """
    mutated_chromosome = []
    PARAMETER_STD_DEV = 0.1  # Standard deviation for parameter mutations

    for layer in chromosome:
        # Optionally delete the entire layer
        if np.random.rand() < LAYER_DELETION_RATE:
            continue
        else:
            new_layer = layer[:]  # Make a shallow copy for mutation
            for j in range(len(new_layer)):
                if new_layer[j] == "-":
                    continue
                elif np.random.rand() < GATE_MUTATION_RATE:
                    gate_type = np.random.choice(["single", "double", "triple", "remove"])
                    if gate_type == "remove":
                        new_layer[j] = "w"
                    elif gate_type == "single":
                        gate_choice = np.random.choice(SINGLE_QUBIT_GATES)
                        if gate_choice in PARAMETRISED_GATES:
                            new_layer[j] = f"{gate_choice}({np.random.normal(0.5, PARAMETER_STD_DEV)})"
                        else:
                            new_layer[j] = f"{gate_choice}"
                    elif gate_type == "double":
                        target_qubit = np.random.randint(0, len(new_layer))
                        control_qubit = np.random.randint(0, len(new_layer))
                        while control_qubit == target_qubit:
                            control_qubit = np.random.randint(0, len(new_layer))
                        gate_choice = np.random.choice(DOUBLE_QUBIT_GATES)
                        if gate_choice in PARAMETRISED_GATES:
                            new_layer[target_qubit] = f"{gate_choice}({control_qubit},{np.random.normal(0.5, PARAMETER_STD_DEV)})"
                        else:
                            new_layer[target_qubit] = f"{gate_choice}({control_qubit})"
                        new_layer[control_qubit] = "-"
                    elif gate_type == "triple" and len(new_layer) >= 3:
                        qubit_1, qubit_2, qubit_3 = np.random.choice(len(new_layer), size=3, replace=False)
                        gate_choice = np.random.choice(TRIPLE_QUBIT_GATES)
                        if gate_choice in PARAMETRISED_GATES:
                            new_layer[qubit_3] = f"{gate_choice}({qubit_1},{qubit_2},{np.random.normal(0.5, PARAMETER_STD_DEV)})"
                        else:
                            new_layer[qubit_3] = f"{gate_choice}({qubit_1},{qubit_2})"
                        new_layer[qubit_1] = "-"
                        new_layer[qubit_2] = "-"
                # Adjust parameter values for parameterised gates
                match = re.match(r"([a-z]+)\((.*)\)", new_layer[j])
                if match and match.group(1) in PARAMETRISED_GATES:
                    gate_name, params_str = match.groups()
                    params_list = [p.strip() for p in params_str.split(",")]
                    # For single-qubit parameterised gates there is only one float parameter.
                    # For double/triple parameterised gates, the last element is the parameter.
                    if len(params_list) >= 1:
                        try:
                            param_value = float(params_list[-1])
                            param_value += np.random.normal(0, PARAMETER_STD_DEV)
                            param_value = max(param_value, 0.0)
                            params_list[-1] = str(param_value)
                        except ValueError:
                            pass
                    new_layer[j] = f"{gate_name}({','.join(params_list)})"
            mutated_chromosome.append(new_layer)
    
    if np.random.rand() < LAYER_MUTATION_RATE:
        new_layer = create_new_layer(len(chromosome[0]))
        mutated_chromosome.append(new_layer)
    
    if not mutated_chromosome:
        mutated_chromosome = initialize_chromosomes(len(chromosome[0]))[0]
    
    return mutated_chromosome

def create_new_layer(qubits):
    """
    Create a new layer for a chromosome with one decision per qubit.
    The new representation:
      - Single-qubit gates are stored without the qubit index.
      - Multi-qubit gates (double or triple) include only the control qubit(s) and any parameters.
      - The target qubit is inferred from the qubit's position.
      - "-" is assigned to qubits that serve as controls.
    """
    available_gates = ["w"] + SINGLE_QUBIT_GATES + DOUBLE_QUBIT_GATES + TRIPLE_QUBIT_GATES
    n_options = len(available_gates)
    weight_barrier = 0.25
    weight_other = 0.75 / (n_options - 1) if n_options > 1 else 0
    weights = [weight_barrier] + [weight_other] * (n_options - 1)
    
    layer = [None] * qubits
    free_indices = list(range(qubits))
    
    while free_indices:
        idx = free_indices.pop(0)
        if layer[idx] is not None:
            continue
        chosen_gate = random.choices(available_gates, weights=weights, k=1)[0]
        
        if chosen_gate in DOUBLE_QUBIT_GATES:
            if free_indices:
                partner = random.choice(free_indices)
                free_indices.remove(partner)
                if chosen_gate in PARAMETRISED_GATES:
                    layer[idx] = f"{chosen_gate}({partner},{np.random.random()})"
                else:
                    layer[idx] = f"{chosen_gate}({partner})"
                if layer[partner] is None:
                    layer[partner] = "-"
            else:
                layer[idx] = "w"
        elif chosen_gate in TRIPLE_QUBIT_GATES:
            if len(free_indices) >= 2:
                partners = random.sample(free_indices, 2)
                for p in partners:
                    free_indices.remove(p)
                if chosen_gate in PARAMETRISED_GATES:
                    layer[idx] = f"{chosen_gate}({partners[0]},{partners[1]},{np.random.random()})"
                else:
                    layer[idx] = f"{chosen_gate}({partners[0]},{partners[1]})"
                if layer[partners[0]] is None:
                    layer[partners[0]] = "-"
                if layer[partners[1]] is None:
                    layer[partners[1]] = "-"
            else:
                layer[idx] = "w"
        else:
            if chosen_gate == "w":
                layer[idx] = "w"
            elif chosen_gate in PARAMETRISED_GATES:
                layer[idx] = f"{chosen_gate}({np.random.random()})"
            else:
                layer[idx] = f"{chosen_gate}"
    
    return layer

# ---------------------------
# Fixer Function for Control Placeholders
# ---------------------------
def fix_control_placeholders(chromosome):
    """
    Fix chromosome layers by replacing unnecessary '-' with 'w'.
    In a given layer, a '-' is only valid if its index is recorded as a control
    for a multi-qubit gate in that layer. Otherwise, it is replaced by a "w"
    (denoting no operation).
    """
    for layer in chromosome:
        used_as_control = set()
        # First pass: identify all valid control qubit indices in this layer.
        for idx, gate_spec in enumerate(layer):
            if gate_spec not in ("-", "w"):
                gate, args = parse_gate_spec(gate_spec)
                if gate in DOUBLE_QUBIT_GATES and args:
                    used_as_control.add(args[0])
                elif gate in TRIPLE_QUBIT_GATES and args:
                    used_as_control.update(args[:2])  # two control qubits
        # Second pass: replace any "-" that is not a valid control with "w"
        for idx, val in enumerate(layer):
            if val == "-" and idx not in used_as_control:
                layer[idx] = "w"
    return chromosome
