# Imports
# -----------------------------------------------------
from qiskit import QuantumCircuit
import re
import np


# Main
# -----------------------------------------------------
def main():
    qubits = 3
    max_circuit_depth = 10
    population = 3
    iterations = 10

    chromosomes = [[["w" for i in range(qubits)] for i in range(max_circuit_depth)] for i in range(population)]
    chromosomes[1][0][0] = "h"

    for i in range(iterations):
        print("------------------POPULATION", str(i) + "------------------")
        circuits = get_circuits(chromosomes)
        fitnesses = get_circuit_fitnesses(circuits)
        child_chromosomes = apply_genetic_operators(chromosomes, fitnesses)
        chromosomes = replace_population(chromosomes, child_chromosomes)


# Representaion
# -----------------------------------------------------
def get_circuits(circuit_chromosomes):
    circuits = []
    
    for circuit_chromosome in circuit_chromosomes:
        # Initialize circuit
        circuit = QuantumCircuit(len(circuit_chromosome[0]))

        # Gate map for Qiskit Aer native gates with explanations
        chromosome_qiskit_gate_map = {
            "w": circuit.barrier,  # Barrier (used for blank "wires")
            "-": None,  # Placeholder for control qubits (no operation)
            "x": circuit.x,  # Pauli-X (NOT) gate
            "y": circuit.y,  # Pauli-Y gate
            "z": circuit.z,  # Pauli-Z gate
            "h": circuit.h,  # Hadamard gate
            "s": circuit.s,  # S (Phase) gate: R_z(π/2)
            "sdg": circuit.sdg,  # S-dagger (Inverse Phase) gate: R_z(-π/2)
            "t": circuit.t,  # T gate: R_z(π/4)
            "tdg": circuit.tdg,  # T-dagger gate: R_z(-π/4)
            "u": circuit.u,  # Generalized single-qubit rotation: R_z(λ) R_y(θ) R_z(φ)
            "rx": circuit.rx,  # Rotation around the X axis: R_x(θ)
            "ry": circuit.ry,  # Rotation around the Y axis: R_y(θ)
            "rz": circuit.rz,  # Rotation around the Z axis: R_z(θ)
            "cx": circuit.cx,  # CNOT (Controlled-X) gate
            "cz": circuit.cz,  # Controlled-Z gate
            "swap": circuit.swap,  # SWAP gate (exchange qubits)
            "ccx": circuit.ccx,  # Toffoli gate (Controlled-Controlled-X)
            "cswap": circuit.cswap,  # Controlled-SWAP gate
            "crx": circuit.crx,  # Controlled-RX rotation gate
            "cry": circuit.cry,  # Controlled-RY rotation gate
            "crz": circuit.crz,  # Controlled-RZ rotation gate
            "cp": circuit.cp,  # Controlled-Phase gate
            "rxx": circuit.rxx,  # Ising interaction: R_xx(θ) (rotation on the XX interaction)
            "ryy": circuit.ryy,  # Ising interaction: R_yy(θ) (rotation on the YY interaction)
            "rzz": circuit.rzz,  # Ising interaction: R_zz(θ) (rotation on the ZZ interaction)
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

        circuits.append(circuit.copy())

    return circuits


# Fitness Function
# -----------------------------------------------------
def get_circuit_fitnesses(circuits):
    fitnesses = [0 for i in range(len(circuits))]

    for circuit in circuits:
        print(circuit)

        value_to_encode = 7

        binary_basis_value = bin(value_to_encode)[2:].zfill(3)
        starting_state = []
        for i in binary_basis_value:
            if i == "1":
                starting_state.append('x')
            else:
                starting_state.append('w')
        print(starting_state)

    return fitnesses


# Genetic Operators
# -----------------------------------------------------
def apply_genetic_operators(chromosomes, fitnesses):
    pass

def mutate_chromosome(chromosome, mutation_rate=0.1):
    """Mutates a single chromosome with a given mutation rate."""
    for layer in chromosome:
        for i in range(len(layer)):
            if np.random.rand() < mutation_rate:
                layer[i] = np.random.choice(["x", "h", "cx(0,1)", "w"])
    return chromosome

def crossover(parent1, parent2):
    """Single-point crossover between two chromosomes."""
    crossover_point = np.random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def replace_population(chromosomes, child_chromosomes):
    return chromosomes

if __name__ == "__main__":
    main()