"""
Refactored simple_optimiser.py

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
import copy
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
from qiskit.quantum_info import Statevector, state_fidelity

# ---------------------------
# Global Constants
# ---------------------------
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
SIMULATOR = AerSimulator(method='statevector')


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
            return copy.deepcopy(chromosomes[idx])
    
    # Fallback (rarely triggered due to float rounding):
    return copy.deepcopy(chromosomes[sorted_indices[-1]])


# ---------------------------
# Initialization and Evaluation Functions
# ---------------------------
def initialize_chromosomes(population, qubits, initial_circuit_depth):
    """
    Initialize a population of chromosomes.

    Each chromosome is a list of layers, where each layer is a list of gate specification strings.

    Parameters:
        population (int): Number of chromosomes.
        qubits (int): Number of qubits (i.e. gates per layer).
        initial_circuit_depth (int): Number of layers per chromosome.

    Returns:
        list: List of chromosomes.
    """
    return [[create_new_layer(qubits) for _ in range(initial_circuit_depth)]
            for _ in range(population)]


def evaluate_random_circuits(population, iterations, qubits, initial_circuit_depth, initial_states, target_states):
    """
    Evaluate a set of randomly generated circuits.

    Parameters:
        population (int): Number of circuits to evaluate per iteration.
        iterations (int): Number of iterations.
        qubits (int): Number of qubits.
        initial_circuit_depth (int): Circuit depth.
        initial_states (list): List of initial state Statevectors.
        target_states (list): List of target state Statevectors (from QFT).

    Returns:
        tuple: (max_fitness, average_fitness) over the evaluated circuits.
    """
    random_circuits = [[create_new_layer(qubits) for _ in range(initial_circuit_depth)]
                       for _ in range(population * iterations)]
    circuits = get_circuits(random_circuits)
    fitnesses = get_circuit_fitnesses(target_states, circuits, initial_states)
    return max(fitnesses), sum(fitnesses) / len(fitnesses)


def get_qft_target_states(qubits, simulator=SIMULATOR):
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
        target_circuit.save_statevector()
        result = SIMULATOR.run(target_circuit).result()
        target_states.append(result.get_statevector())
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
def get_circuit_fitnesses(target_states, circuits, initial_states, simulator=SIMULATOR):
    """
    Evaluate the fitness of each circuit based on similarity to the QFT target states.

    The fitness of a circuit is computed by initializing it with every initial state and comparing
    the resulting statevector to the corresponding target state via state fidelity.

    Parameters:
        target_states (list): List of target statevectors.
        circuits (list): List of QuantumCircuit objects.
        initial_states (list): List of initial state Statevectors.
        simulator (AerSimulator): The Qiskit simulator.

    Returns:
        list: List of fitness values (one per circuit).
    """
    fitnesses = []
    batch_circuits = []
    for circuit in circuits:
        for state in initial_states:
            new_circuit = QuantumCircuit(*circuit.qregs)
            new_circuit.initialize(state)
            new_circuit.compose(circuit, inplace=True)
            new_circuit.save_statevector()
            batch_circuits.append(new_circuit)
    results = simulator.run(batch_circuits).result()
    output_states = [results.get_statevector(i) for i in range(len(batch_circuits))]
    
    group_size = len(initial_states)
    for i in range(0, len(output_states), group_size):
        circuit_states = output_states[i:i + group_size]
        fitnesses.append(compute_phase_fitness(circuit_states, target_states))
    return fitnesses


def compute_phase_fitness(circuit_states, target_states):
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


def phase_sensitive_fidelity(output_state, target_state):
    """
    Compute a phase-sensitive fidelity metric between two statevectors.

    Parameters:
        output_state (Statevector): The state produced by the circuit.
        target_state (Statevector): The target state.

    Returns:
        float: Phase-sensitive fidelity.
    """
    output_sv = Statevector(output_state)
    target_sv = Statevector(target_state)
    output_phases = np.angle(output_sv.data)
    target_phases = np.angle(target_sv.data)
    phase_differences = (output_phases - target_phases) % (2 * np.pi)
    phase_differences = np.where(phase_differences > np.pi, phase_differences - 2 * np.pi, phase_differences)
    return 1 - np.mean(phase_differences ** 2) / (np.pi ** 2)


# ---------------------------
# Genetic Operators
# ---------------------------
def apply_genetic_operators(
    chromosomes, fitnesses, elite_count, parameter_mutation_rate, gate_mutation_rate,
    layer_mutation_rate, max_parameter_mutation, layer_deletion_rate
):
    """
    Creates a new population by preserving 'elite_count' top individuals,
    then using rank selection to fill the rest.
    """
    # Identify elites
    sorted_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)
    elites = [copy.deepcopy(chromosomes[i]) for i in sorted_indices[:elite_count]]
    
    # Generate the remaining population with rank selection
    new_population = []
    remaining = len(chromosomes) - elite_count
    while len(new_population) < remaining:
        parent1 = rank_selection(chromosomes, fitnesses)
        parent2 = rank_selection(chromosomes, fitnesses)
        child1, child2 = crossover(parent1, parent2)
        
        child1 = mutate_chromosome(
            child1, parameter_mutation_rate, gate_mutation_rate,
            layer_mutation_rate, max_parameter_mutation, layer_deletion_rate
        )
        child2 = mutate_chromosome(
            child2, parameter_mutation_rate, gate_mutation_rate,
            layer_mutation_rate, max_parameter_mutation, layer_deletion_rate
        )
        
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
    crossover_point = np.random.randint(1, len(parent_1))
    child_1 = parent_1[:crossover_point] + parent_2[crossover_point:]
    child_2 = parent_2[:crossover_point] + parent_1[crossover_point:]
    return child_1, child_2


def mutate_chromosome(chromosome, parameter_mutation_rate, gate_mutation_rate,
                      layer_mutation_rate, max_parameter_mutation, layer_deletion_rate):
    """
    Mutate a single chromosome by modifying gate types, parameters, or layers.

    Parameters:
        chromosome (list): The chromosome to mutate.
        parameter_mutation_rate (float): Rate of parameter mutation.
        gate_mutation_rate (float): Rate of gate type mutation.
        layer_mutation_rate (float): Probability of adding a new layer.
        max_parameter_mutation (float): Maximum factor for parameter mutation.
        layer_deletion_rate (float): Probability of deleting an entire layer.

    Returns:
        list: Mutated chromosome.
    """
    for i, layer in enumerate(chromosome):
        if np.random.rand() < layer_deletion_rate:
            chromosome[i] = ["w"] * len(layer)
            continue
        for j in range(len(layer)):
            if np.random.rand() < gate_mutation_rate:
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
            elif np.random.rand() < parameter_mutation_rate:
                match = re.match(r"([a-z]+)\((.*)\)", layer[j])
                if match and match.group(1) in PARAMETRISED_GATES:
                    gate_name, params_str = match.groups()
                    params_list = [p.strip() for p in params_str.split(",")]
                    param_value = float(params_list[-1])
                    factor = np.random.uniform(1, max_parameter_mutation)
                    if np.random.rand() < 0.5:
                        param_value *= factor
                    else:
                        param_value /= factor
                    param_value = max(param_value, 0.0)
                    params_list[-1] = str(param_value)
                    layer[j] = f"{gate_name}({','.join(params_list)})"
    if np.random.rand() < layer_mutation_rate:
        new_layer = create_new_layer(len(chromosome[0]))
        chromosome.append(new_layer)
    return chromosome


def create_new_layer(qubits):
    """
    Create a new layer for a chromosome.

    Parameters:
        qubits (int): Number of qubits (gates in the layer).

    Returns:
        list: A new layer represented as a list of gate specification strings.
    """
    layer = ["w"] * qubits
    for qubit in range(qubits):
        if np.random.rand() < 0.7:
            layer[qubit] = "w"
        else:
            gate_choice = np.random.choice(SINGLE_QUBIT_GATES)
            if gate_choice in PARAMETRISED_GATES:
                layer[qubit] = f"{gate_choice}({qubit}, {np.random.random()})"
            else:
                layer[qubit] = f"{gate_choice}({qubit})"
    if np.random.rand() < 0.5:
        target_qubit = np.random.randint(0, qubits)
        control_qubit = np.random.randint(0, qubits)
        while control_qubit == target_qubit:
            control_qubit = np.random.randint(0, qubits)
        gate_choice = np.random.choice(DOUBLE_QUBIT_GATES)
        if gate_choice in PARAMETRISED_GATES:
            layer[target_qubit] = f"{gate_choice}({control_qubit},{target_qubit},{np.random.random()})"
        else:
            layer[target_qubit] = f"{gate_choice}({control_qubit},{target_qubit})"
        layer[control_qubit] = "-"
    if np.random.rand() < 0.2:
        qubit_1, qubit_2, qubit_3 = np.random.choice(range(qubits), size=3, replace=False)
        gate_choice = np.random.choice(TRIPLE_QUBIT_GATES)
        if gate_choice in PARAMETRISED_GATES:
            layer[qubit_3] = f"{gate_choice}({qubit_1},{qubit_2},{qubit_3},{np.random.random()})"
        else:
            layer[qubit_3] = f"{gate_choice}({qubit_1},{qubit_2},{qubit_3})"
        layer[qubit_1] = "-"
        layer[qubit_2] = "-"
    return layer
