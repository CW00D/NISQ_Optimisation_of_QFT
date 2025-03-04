import ast
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit.quantum_info import state_fidelity
from qiskit.circuit.library import QFT
from qiskit_ibm_runtime import QiskitRuntimeService
# Import your circuit conversion function from your EA module.
from optimiser_noisy import get_circuits

# ---------------------------
# STEP 1: Load the CSV file with circuit chromosomes
# ---------------------------
csv_file_path = r"Experiment Results\\Chromosomes\\Optimiser_noisy\\2025-03-03_19-34.csv"
df = pd.read_csv(csv_file_path)
df['Chromosome'] = df['Chromosome'].apply(ast.literal_eval)
top_chromosomes = df['Chromosome'].tolist()
print("Loaded the top chromosomes from the CSV file.")

# ---------------------------
# STEP 2: Convert chromosomes to QuantumCircuit objects
# ---------------------------
circuits = get_circuits(top_chromosomes)
# Assume all chromosomes have the same number of qubits:
num_qubits = len(top_chromosomes[0][0])
print(f"Converted the top chromosomes to QuantumCircuit objects with {num_qubits} qubits.")

# ---------------------------
# STEP 3: Set up Simulators (Noiseless & Noisy)
# ---------------------------
service = QiskitRuntimeService()
backend = service.backend('ibm_brisbane')
noise_model = NoiseModel.from_backend(backend)
noiseless_simulator = AerSimulator(method='density_matrix')
noisy_simulator = AerSimulator(method='density_matrix', noise_model=noise_model)
print("Set up the noiseless and noisy simulators.")

# ---------------------------
# STEP 4: Define Helper Functions for Evaluation
# ---------------------------
native_gates = AerSimulator(method='density_matrix').configuration().basis_gates

def get_qft_target_states(qubits, simulator):
    target_states = []
    for i in range(2 ** qubits):
        qc = QuantumCircuit(qubits)
        binary_str = format(i, f'0{qubits}b')
        for j, bit in enumerate(binary_str):
            if bit == '1':
                qc.x(j)
        qc.append(QFT(num_qubits=qubits), list(range(qubits)))
        qc = transpile(qc, basis_gates=['u', 'cx'])
        qc.save_density_matrix()
        result = simulator.run(qc).result()
        density_matrix = result.data(0)['density_matrix']
        target_states.append(density_matrix)
    return target_states

def evaluate_circuit_fitness(circuit, simulator, num_qubits, target_states):
    fitness_total = 0.0
    for i, target_state in enumerate(target_states):
        qc = QuantumCircuit(num_qubits)
        binary_str = format(i, f'0{num_qubits}b')
        for j, bit in enumerate(binary_str):
            if bit == '1':
                qc.x(j)
        transpiled_circuit = transpile(circuit, basis_gates=native_gates)
        qc.compose(transpiled_circuit, inplace=True)
        qc.save_density_matrix()
        result = simulator.run(qc).result()
        state = result.data(0)['density_matrix']
        fidelity = state_fidelity(state, target_state)
        fitness_total += fidelity
        print("Evaluated the circuit's fitness under both conditions.")
    return fitness_total / (2 ** num_qubits)
    
# ---------------------------
# STEP 5: Generate Target Density Matrices
# ---------------------------
target_states = get_qft_target_states(num_qubits, noiseless_simulator)
print("Generated the target density matrices for the QFT circuit.")

# ---------------------------
# STEP 6: Evaluate Each Circuit Under Both Conditions
# ---------------------------
noiseless_fidelities = [evaluate_circuit_fitness(circ, noiseless_simulator, num_qubits, target_states)
                        for circ in circuits]
noisy_fidelities = [evaluate_circuit_fitness(circ, noisy_simulator, num_qubits, target_states)
                    for circ in circuits]

results = pd.DataFrame({
    "Circuit Number": list(range(1, len(circuits) + 1)),
    "Noiseless Fidelity": noiseless_fidelities,
    "Noisy Fidelity": noisy_fidelities
})
results["% Drop"] = ((results["Noiseless Fidelity"] - results["Noisy Fidelity"]) /
                     results["Noiseless Fidelity"]) * 100
print("Evaluated each circuit's fitness under both conditions.")

# ---------------------------
# STEP 7: Evaluate the QFT Circuit Itself
# ---------------------------
qft_circuit = QuantumCircuit(num_qubits)
qft_circuit.append(QFT(num_qubits=num_qubits), list(range(num_qubits)))
qft_noiseless = evaluate_circuit_fitness(qft_circuit, noiseless_simulator, num_qubits, target_states)
qft_noisy = evaluate_circuit_fitness(qft_circuit, noisy_simulator, num_qubits, target_states)
qft_drop = ((qft_noiseless - qft_noisy) / qft_noiseless) * 100 if qft_noiseless != 0 else 0

qft_row = {"Circuit Number": "QFT Circuit",
           "Noiseless Fidelity": qft_noiseless,
           "Noisy Fidelity": qft_noisy,
           "% Drop": qft_drop}

results = pd.concat([pd.DataFrame([qft_row]), results], ignore_index=True)
print("Evaluated the QFT circuit's fitness under both conditions.")

# ---------------------------
# STEP 8: Create and Plot the Results Table with Rounded Values
# ---------------------------
# Create a copy for display and round numeric columns to 10 decimal places.
display_df = results.copy()
for col in ["Noiseless Fidelity", "Noisy Fidelity", "% Drop"]:
    display_df[col] = display_df[col].apply(lambda x: round(x, 10) if isinstance(x, (int, float)) else x)

fig, ax = plt.subplots(figsize=(8, (len(display_df))*0.5 + 1))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=display_df.values,
                 colLabels=display_df.columns,
                 cellLoc='center',
                 loc='center')

# Highlight the QFT Circuit row in blue (assumed at table row index 1)
for (row, col), cell in table.get_celld().items():
    if row == 1:
        cell.set_facecolor('#add8e6')

plt.title("Circuit Performance: Noiseless vs Noisy Execution")
plt.show()
