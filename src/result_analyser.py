import ast
import pandas as pd
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

import numpy as np
import os

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error, phase_damping_error
from qiskit.quantum_info import state_fidelity
from qiskit.circuit.library import QFT
from optimiser_noisy import get_circuits

from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService

# Load IBM Quantum account
service = QiskitRuntimeService()
print("Loaded backend models")

def evaluate_noisy_fitness_multiple_times(circuit, simulator, num_qubits, target_states, num_evaluations=1):
    total_fitness = 0.0
    for _ in range(num_evaluations):
        total_fitness += evaluate_circuit_fitness(circuit, simulator, num_qubits, target_states)
    return total_fitness / num_evaluations

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
    return fitness_total / (2 ** num_qubits)

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
amp_gamma = 0.002    # 0.2% amplitude damping error
phase_gamma = 0.002  # 0.2% phase damping error

# Create a new noise model instance for the training noise model
noise_model = NoiseModel()

# Apply single-qubit errors (combine depolarizing, amplitude damping, and phase damping)
for gate, rate in single_qubit_error_rates.items():
    if rate > 0:
        error_depol = depolarizing_error(rate, 1)
        error_amp = amplitude_damping_error(amp_gamma)
        error_phase = phase_damping_error(phase_gamma)
        combined_error = error_depol.compose(error_amp).compose(error_phase)
        noise_model.add_all_qubit_quantum_error(combined_error, gate)

# Apply two-qubit errors using only depolarizing noise
for gate, rate in two_qubit_error_rates.items():
    error_2q = depolarizing_error(rate, 2)
    noise_model.add_all_qubit_quantum_error(error_2q, gate)

# Apply three-qubit errors using depolarizing noise
for gate, rate in three_qubit_error_rates.items():
    error_3q = depolarizing_error(rate, 3)
    noise_model.add_all_qubit_quantum_error(error_3q, gate)

# Set up simulators
noiseless_simulator = AerSimulator(method='density_matrix')
noisy_simulator = AerSimulator(method='density_matrix', noise_model=noise_model)
print("Set up the noiseless and trained noisy simulators.")

# Get the native basis gates for transpilation from the default simulator configuration
native_gates = AerSimulator(method='density_matrix').configuration().basis_gates

# -----------------------------
# Set up the unseen noise model using IBM's ibm_brisbane
# (We now ignore unseen fidelity results.)
backend = service.backend('ibm_brisbane')
noise_model_unseen = NoiseModel.from_backend(backend)
unseen_simulator = AerSimulator(method='density_matrix', noise_model=noise_model_unseen)
print("Set up the unseen simulator using the ibm_brisbane noise model.")
# -----------------------------

# Directories to process
num_qubits = 2  # Adjust as needed
directories = [
    r"Experiment Results\\Optimiser_simple\\" + str(num_qubits) + " Qubit Simulation\Data",
    r"Experiment Results\\Optimiser_depth_reduction\\" + str(num_qubits) + " Qubit Simulation\Data",
    r"Experiment Results\\Optimiser_noisy\\" + str(num_qubits) + " Qubit Simulation\Data",
    r"Experiment Results\\Optimiser_noisy_depth_reduction\\" + str(num_qubits) + " Qubit Simulation\Data"
]

results_list = []
print("Processing the directories...")

for directory in directories:
    simulation_name = os.path.basename(os.path.dirname(os.path.dirname(directory)))
    run_files = [f for f in os.listdir(directory) if f.endswith("_final_chromosomes.csv")]
    
    for run_file in run_files:
        csv_file_path = os.path.join(directory, run_file)
        df = pd.read_csv(csv_file_path)
        df['Chromosome'] = df['Chromosome'].apply(ast.literal_eval)
        top_chromosomes = df['Chromosome'].tolist()[:3]  # Top 3 chromosomes
        
        circuits = get_circuits(top_chromosomes)
        num_qubits = len(top_chromosomes[0][0])
        target_states = get_qft_target_states(num_qubits, noiseless_simulator)
        
        noiseless_fidelities = [evaluate_circuit_fitness(circ, noiseless_simulator, num_qubits, target_states)
                                for circ in circuits]
        noisy_fidelities = [evaluate_noisy_fitness_multiple_times(circ, noisy_simulator, num_qubits, target_states)
                            for circ in circuits]
        
        avg_noiseless_fidelity = np.mean(noiseless_fidelities)
        avg_noisy_fidelity = np.mean(noisy_fidelities)
        
        avg_fitness_drop = np.mean([(nf - nf_noisy) / nf * 100 for nf, nf_noisy in zip(noiseless_fidelities, noisy_fidelities)])
        
        results_list.append({
            "Simulation Type": simulation_name,
            "Run": run_file.split('_')[0],
            "Avg Noiseless Fidelity": avg_noiseless_fidelity,
            "Avg Noisy Fidelity": avg_noisy_fidelity,
            "% Fidelity Drop": avg_fitness_drop
        })

# Calculate traditional QFT performance (only one circuit, so one row)
qft_circuit = QFT(num_qubits)
target_states = get_qft_target_states(num_qubits, noiseless_simulator)
noiseless_fidelity_qft = evaluate_circuit_fitness(qft_circuit, noiseless_simulator, num_qubits, target_states)
noisy_fidelity_qft = evaluate_noisy_fitness_multiple_times(qft_circuit, noisy_simulator, num_qubits, target_states)
fitness_drop_qft = (noiseless_fidelity_qft - noisy_fidelity_qft) / noiseless_fidelity_qft * 100

# Insert Traditional QFT result as a single row
results_list.insert(0, {
    "Simulation Type": "Traditional QFT",
    "Run": "Run",
    "Avg Noiseless Fidelity": noiseless_fidelity_qft,
    "Avg Noisy Fidelity": noisy_fidelity_qft,
    "% Fidelity Drop": fitness_drop_qft
})

results_df = pd.DataFrame(results_list)

# Set the desired order for simulation types
simulation_type_order = [
    "Traditional QFT", "Optimiser_simple", "Optimiser_depth_reduction",
    "Optimiser_noisy", "Optimiser_noisy_depth_reduction"
]
results_df["Simulation Type"] = pd.Categorical(results_df["Simulation Type"],
                                               categories=simulation_type_order,
                                               ordered=True)
results_df.sort_values(["Simulation Type", "Run"], inplace=True)

# --- Create Aggregated Summary ---
# For each simulation type (except QFT), add two rows: one for the average and one for the maximum.
summary_list = []
for sim_type in simulation_type_order:
    if sim_type == "Traditional QFT":
        # Keep the QFT row as-is.
        qft_row = results_df[results_df["Simulation Type"] == sim_type].iloc[0].to_dict()
        summary_list.append(qft_row)
    else:
        group = results_df[results_df["Simulation Type"] == sim_type]
        if not group.empty:
            avg_row = {
                "Simulation Type": sim_type,
                "Run": "Average",
                "Avg Noiseless Fidelity": group["Avg Noiseless Fidelity"].mean(),
                "Avg Noisy Fidelity": group["Avg Noisy Fidelity"].mean(),
                "Std Dev Noisy Fidelity": group["Avg Noisy Fidelity"].std(),  # Compute standard deviation
                "% Fidelity Drop": group["% Fidelity Drop"].mean()
            }
            max_index = group["Avg Noisy Fidelity"].idxmax()
            max_row = group.loc[max_index].to_dict()
            max_row["Run"] = "Best Circuit"
            max_row["Std Dev Noisy Fidelity"] = "-"
            summary_list.append(avg_row)
            summary_list.append(max_row)

summary_df = pd.DataFrame(summary_list)
summary_df = summary_df.round(6)

# --- Plotting the Aggregated Summary Table ---
fig, ax = plt.subplots(figsize=(12, len(summary_df) * 0.7 + 1))
ax.axis('tight')
ax.axis('off')

# Create the table using the aggregated summary DataFrame
table = ax.table(cellText=summary_df.values,
                 colLabels=summary_df.columns,
                 cellLoc='center',
                 loc='center',
                 bbox=[0, 0, 1, 1])

# Adjust column widths
for key, cell in table.get_celld().items():
    cell.set_width(0.15)
    # Increase font size
    cell.set_fontsize(20)

# Style header cells (bold and increased font size)
for key, cell in table.get_celld().items():
    if key[0] == 0:
        cell.set_facecolor('#000080')
        cell.set_text_props(color='white', weight='bold', fontsize=20)

# Apply row shading:
# - Highlight Traditional QFT row in light green.
# - Shade Optimiser_depth_reduction and Optimiser_noisy_depth_reduction rows in grey.
n_rows = summary_df.shape[0]
for i in range(1, n_rows+1):  # table rows (header is row 0)
    sim_type = table[(i, 0)].get_text().get_text()
    if sim_type == "Traditional QFT":
        for j in range(summary_df.shape[1]):
            table[(i, j)].set_facecolor('#d9ead3')  # light green
    elif sim_type in ["Optimiser_depth_reduction", "Optimiser_noisy_depth_reduction"]:
        for j in range(summary_df.shape[1]):
            table[(i, j)].set_facecolor('#d9d9d9')  # grey

# --- Save the Plot --- 
output_dir = "Experiment Results/Summary Plots"
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

output_filepath = os.path.join(output_dir, f"summary_plot_{num_qubits}_qubits.png")
plt.title("Fitnesses Performance Analysis", fontsize=20)
fig.savefig(output_filepath, bbox_inches='tight', dpi=300)
print(f"Plot saved to: {output_filepath}")
plt.show()
