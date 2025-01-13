# Imports
# -----------------------------------------------------
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QFT
from qiskit.quantum_info import Statevector, state_fidelity
import re
import numpy as np
import copy


# Main
# -----------------------------------------------------
def main():
    qubits = 3
    initial_circuit_depth = 10
    population = 20
    iterations = 1000

    possible_starting_gates = ["w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "w", "x", "y", "z", "h", "s", "sdg", "t", "tdg"]
    chromosomes = [[[possible_starting_gates[np.random.choice(len(possible_starting_gates))] for i in range(qubits)] for i in range(initial_circuit_depth)] for i in range(population)]
    chromosomes[1][0][0] = "h"

    for i in range(iterations):
        #print("\n\n------------------POPULATION", str(i) + "------------------\n")
        circuits = get_circuits(chromosomes)
        fitnesses = get_circuit_fitnesses(circuits, qubits)
        print(max(fitnesses))
        max_fitness_chromosome = chromosomes[max(range(len(fitnesses)), key=fitnesses.__getitem__)]
        if max(fitnesses) >= 0.999:
            print("Stopping early: Fitness threshold reached")
            break
        chromosomes = apply_genetic_operators(chromosomes, fitnesses)

    print(max_fitness_chromosome)
    print(get_circuits([max_fitness_chromosome])[0])

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
            "-": lambda qubit: None,  # Placeholder for control qubits (no operation)
            "x": lambda qubit: circuit.x(qubit),  # Pauli-X (NOT) gate
            "y": lambda qubit: circuit.y(qubit),  # Pauli-Y gate
            "z": lambda qubit: circuit.z(qubit),  # Pauli-Z gate
            "h": lambda qubit: circuit.h(qubit),  # Hadamard gate
            "s": lambda qubit: circuit.s(qubit),  # S (Phase) gate: R_z(π/2)
            "sdg": lambda qubit: circuit.sdg(qubit),  # S-dagger (Inverse Phase) gate: R_z(-π/2)
            "t": lambda qubit: circuit.t(qubit),  # T gate: R_z(π/4)
            "tdg": lambda qubit: circuit.tdg(qubit),  # T-dagger gate: R_z(-π/4)
            "u": lambda qubit: circuit.u(qubit),  # Generalised single-qubit rotation: R_z(λ) R_y(θ) R_z(φ)
            "rx": lambda qubit: circuit.rx(qubit),  # Rotation around the X axis: R_x(θ)
            "ry": lambda qubit: circuit.ry(qubit),  # Rotation around the Y axis: R_y(θ)
            "rz": lambda qubit: circuit.rz(qubit),  # Rotation around the Z axis: R_z(θ)
            "cx": lambda qubit: circuit.cx(qubit),  # CNOT (Controlled-X) gate
            "cz": lambda qubit: circuit.cz(qubit),  # Controlled-Z gate
            "swap": lambda qubit: circuit.swap(qubit),  # SWAP gate (exchange qubits)
            "ccx": lambda qubit: circuit.ccx(qubit),  # Toffoli gate (Controlled-Controlled-X)
            "cswap": lambda qubit: circuit.cswap(qubit),  # Controlled-SWAP gate
            "crx": lambda qubit: circuit.crx(qubit),  # Controlled-RX rotation gate
            "cry": lambda qubit: circuit.cry(qubit),  # Controlled-RY rotation gate
            "crz": lambda qubit: circuit.crz(qubit),  # Controlled-RZ rotation gate
            "cp": lambda qubit: circuit.cp(qubit),  # Controlled-Phase gate
            "rxx": lambda qubit: circuit.rxx(qubit),  # Ising interaction: R_xx(θ) (rotation on the XX interaction)
            "ryy": lambda qubit: circuit.ryy(qubit),  # Ising interaction: R_yy(θ) (rotation on the YY interaction)
            "rzz": lambda qubit: circuit.rzz(qubit),  # Ising interaction: R_zz(θ) (rotation on the ZZ interaction)
        }

        # Helper to apply gates
        for block in circuit_chromosome:
            for qubit in range(len(block)):
                gate_spec = block[qubit]
                if gate_spec == "-":
                    continue
                elif "(" in gate_spec:
                    gate, args = re.match(r"(\w+)\((.+)\)", gate_spec).groups()
                    args = list(map(int, args.split(",")))
                    chromosome_qiskit_gate_map[gate](*args)
                else:
                    chromosome_qiskit_gate_map[gate_spec](qubit)
        circuit.save_statevector()

        circuits.append(circuit.copy())
        #print(circuit, "\n")

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
def apply_genetic_operators(chromosomes, fitnesses, parent_chromosomes=10):  
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

def mutate_chromosome(chromosome, mutation_rate=0.1):
    """Mutates a single chromosome with a given mutation rate."""
    for layer in chromosome:
        for i in range(len(layer)):
            if np.random.rand() < mutation_rate:
                layer[i] = np.random.choice(["w", "w", "w", "w", "w", "w", "w", "w", "x", "y", "z", "h", "s", "sdg", "t", "tdg"])
    return chromosome

def replace_population(chromosomes, child_chromosomes, bottom_indices):
    """ Replace bottom chromosomes with children."""
    new_population = chromosomes.copy()
    for i, idx in enumerate(bottom_indices):
        new_population[idx] = child_chromosomes[i]
    return new_population


if __name__ == "__main__":
    main()