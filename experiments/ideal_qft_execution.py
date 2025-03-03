
import optimiser_simple

qft_chromosome = [
        ["h(0)", "w", "w"],  # Hadamard on qubit 0
        ["cp(0,1,1.5707963267948966)", "w", "w"],  # Controlled rotation π/2 between qubits 0 and 1
        ["cp(0,2,0.7853981633974483)", "w", "w"],  # Controlled rotation π/4 between qubits 0 and 2
        ["w", "h(1)", "w"],  # Hadamard on qubit 1
        ["w", "cp(1,2,1.5707963267948966)", "w"],  # Controlled rotation π/2 between qubits 1 and 2
        ["w", "w", "h(2)"],  # Hadamard on qubit 2
        ["swap(0,2)", "w", "w"],  # Swap qubits 0 and 2
    ]
    
qft_circuit = optimiser_simple.get_circuits([qft_chromosome])
fitness = optimiser_simple.get_circuit_fitnesses(qft_circuit, 3)
print(fitness)