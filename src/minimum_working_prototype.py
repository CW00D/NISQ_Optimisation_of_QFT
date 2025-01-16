# Imports
# -----------------------------------------------------
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QFT
from qiskit.quantum_info import Statevector, state_fidelity
import re
import numpy as np
import copy
from numba import njit

# Gate lists
single_qubit_gates = [
    "x",
    "y",
    "z",
    "h",
    "s",
    "sdg",
    "t",
    "tdg",
    #"u",
    "rx",
    "ry",
    "rz"
]

double_qubit_gates = [
    "cx",
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
    "cswap",
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
population = 20
qubits = 3
initial_circuit_depth = 10
population = 20
iterations = 100

# Main
# -----------------------------------------------------
def main():
    # Generate initial chromosomes
    chromosomes = initialize_chromosomes(
        population,
        qubits,
        initial_circuit_depth,
        single_qubit_gates,
        double_qubit_gates,
        triple_qubit_gates
    )

    for i in range(iterations):
        #print("\n\n------------------POPULATION", str(i) + "------------------\n")
        circuits = get_circuits(chromosomes)
        fitnesses = get_circuit_fitnesses(circuits, qubits)
        max_fitness_chromosome = chromosomes[max(range(len(fitnesses)), key=fitnesses.__getitem__)]
        if max(fitnesses) >= 0.999:
            print("Stopping early: Fitness threshold reached")
            break
        chromosomes = apply_genetic_operators(chromosomes, fitnesses)

    print("==================================================")
    print(max(fitnesses))
    print(get_circuits([max_fitness_chromosome])[0])
    print("==================================================")
    circuits = get_circuits(chromosomes)
    for circuit in circuits:
        print(circuit)

def initialize_chromosomes(population, qubits, initial_circuit_depth, single_qubit_gates, double_qubit_gates, triple_qubit_gates):
    chromosomes = []

    for i in range(population):
        chromosome = []

        for i in range(initial_circuit_depth):
            layer = create_new_layer(qubits, single_qubit_gates, double_qubit_gates, triple_qubit_gates)
            chromosome.append(layer)

        chromosomes.append(chromosome)

    return chromosomes


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
            #"u": lambda qubit, params: circuit.u(*params, qubit),  # Generalised single-qubit rotation: R_z(λ) R_y(θ) R_z(φ)
            "rx": lambda qubit, theta: circuit.rx(theta, qubit),  # Rotation around the X axis: R_x(θ)
            "ry": lambda qubit, theta: circuit.ry(theta, qubit),  # Rotation around the Y axis: R_y(θ)
            "rz": lambda qubit, theta: circuit.rz(theta, qubit),  # Rotation around the Z axis: R_z(θ)
            "cx": lambda control_qubit, target_qubit: circuit.cx(control_qubit, target_qubit),  # CNOT (Controlled-X) gate
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
            result = simulator.run(circuit, initial_state=state).result()
            circuit_states.append(result.get_statevector())

        # Compute fitness based on phase differences
        fitness = compute_phase_fitness(circuit_states, target_states)
        fitnesses.append(fitness)

    return fitnesses

def get_qft_target_states(qubits):
    """Simulates the QFT circuit for all basis states and returns the output states."""
    target_states = []
    simulator = AerSimulator(method='statevector')

    for i in range(2**qubits):
        state_binary = f"{i:0{qubits}b}"  # Binary string representing the basis state
        
        # Create a fresh target circuit for each initial state
        target_circuit = QuantumCircuit(qubits)
        
        # Apply X gates to prepare the initial state |state_binary⟩
        for j, bit in enumerate(state_binary):
            if bit == "1":
                target_circuit.x(j)
        
        # Apply the QFT
        target_circuit.append(QFT(num_qubits=qubits), range(qubits))
        target_circuit = transpile(target_circuit, basis_gates=['u', 'cx'])
        target_circuit.save_statevector()
        
        # Simulate the circuit
        result = simulator.run(target_circuit).result()
        target_states.append(result.get_statevector())

    return target_states

def compute_phase_fitness(circuit_states, target_states):
    """Computes fitness based on phase differences."""
    total_phase_error = 0
    for circuit_state, target_state in zip(circuit_states, target_states):
        circuit_phases = get_phase_differences(circuit_state)
        target_phases = get_phase_differences(target_state)
        phase_error = np.linalg.norm(circuit_phases - target_phases)
        total_phase_error += phase_error
    
    # Convert phase error to fitness (lower error means higher fitness)
    max_possible_error = 2 * np.pi * len(circuit_states)
    fitness = 1 - (total_phase_error / max_possible_error)
    return max(fitness, 0)  # Ensure fitness is within [0, 1]

def get_phase_differences(state):
    """Returns the phases of the complex amplitudes in the statevector."""
    return np.angle(state.data)


# Genetic Operators
# -----------------------------------------------------
def apply_genetic_operators(chromosomes, fitnesses, parent_chromosomes=population//2):  
    # Sort chromosomes by fitness in descending order
    sorted_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)

    # Preserve the top `parent_chromosomes` chromosomes (deep copy to avoid mutation)
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
    crossover_point = np.random.randint(1, len(parent_1) - 1)
    child_1 = parent_1[:crossover_point] + parent_2[crossover_point:]
    child_2 = parent_2[:crossover_point] + parent_1[crossover_point:]
    return child_1, child_2

def mutate_chromosome(chromosome, parameter_mutation_rate=0.1, gate_mutation_rate=0.1, layer_mutation_rate=0.1, max_parameter_mutation=0.1):
    """Mutates a single chromosome with a given mutation rate."""

    for layer in chromosome:
        for i in range(len(layer)):
            # Mutate the gate type
            if np.random.rand() < gate_mutation_rate:
                gate_type = np.random.choice(["single", "double", "triple"])
                if gate_type == "single":
                    gate_choice = np.random.choice(single_qubit_gates)
                    if gate_choice in parametrised_gates:
                        layer[i] = gate_choice + f"({i}, {np.random.random()})"
                    else:
                        layer[i] = gate_choice + f"({i})"
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

            # Mutate gate parameters with multiplication or division by a factor
            elif np.random.rand() < parameter_mutation_rate:
                match = re.match(r"([a-z]+)\((.*)\)", layer[i])
                if match and match.group(1) in parametrised_gates:
                    gate_name, params = match.groups()
                    params_list = params.split(",")
                    
                    # Get the parameter that we need to mutate (last element in the parameter list)
                    param_value = float(params_list[-1])
                    
                    # Randomly choose a factor to multiply or divide by, within the max_parameter_mutation range
                    factor = np.random.uniform(1, max_parameter_mutation)
                    if np.random.rand() < 0.5:  # 50% chance for multiplication or division
                        param_value *= factor  # Multiply by the factor
                    else:
                        param_value /= factor  # Divide by the factor

                    # Ensure the parameter stays positive (optional)
                    param_value = max(param_value, 0.0)

                    # Update the mutated parameter in the list
                    params_list[-1] = str(param_value)

                    # Reassemble the gate with the updated parameter
                    layer[i] = f"{gate_name}({','.join(params_list)})"

    # Add a new layer with a certain probability
    if np.random.rand() < layer_mutation_rate:
        new_layer = create_new_layer(len(chromosome[0]), single_qubit_gates, double_qubit_gates, triple_qubit_gates)
        chromosome.append(new_layer)

    return chromosome

def create_new_layer(qubits, single_qubit_gates, double_qubit_gates, triple_qubit_gates):
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


if __name__ == "__main__":
    main()