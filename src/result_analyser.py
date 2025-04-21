import ast
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT
from qiskit.quantum_info import state_fidelity
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error, phase_damping_error
from qiskit_ibm_runtime import QiskitRuntimeService

from optimiser_noisy import get_circuits

# Load IBM Quantum service
service = QiskitRuntimeService()

# Define noise model
single_qubit_error_rates = {
    "x": 0.002, "y": 0.002, "z": 0.001, "h": 0.0015, "s": 0.001, "sdg": 0.001,
    "t": 0.0015, "tdg": 0.0015, "rx": 0.002, "ry": 0.002, "rz": 0.0
}
two_qubit_error_rates = {
    "cx": 0.02, "cy": 0.02, "cz": 0.018, "swap": 0.025,
    "crx": 0.02, "cry": 0.02, "crz": 0.02, "cp": 0.02,
    "rxx": 0.02, "ryy": 0.02, "rzz": 0.02
}
three_qubit_error_rates = {
    "ccx": 0.1,
    "cswap": 0.12
}
amp_gamma = 0.002
phase_gamma = 0.002

noise_model = NoiseModel()
for gate, rate in single_qubit_error_rates.items():
    if rate > 0:
        error = depolarizing_error(rate, 1).compose(amplitude_damping_error(amp_gamma)).compose(phase_damping_error(phase_gamma))
        noise_model.add_all_qubit_quantum_error(error, gate)
for gate, rate in two_qubit_error_rates.items():
    noise_model.add_all_qubit_quantum_error(depolarizing_error(rate, 2), gate)
for gate, rate in three_qubit_error_rates.items():
    noise_model.add_all_qubit_quantum_error(depolarizing_error(rate, 3), gate)

# Simulators
noiseless_simulator = AerSimulator(method='density_matrix')
noisy_simulator = AerSimulator(method='density_matrix', noise_model=noise_model)
native_gates = noiseless_simulator.configuration().basis_gates

def get_qft_target_states(qubits, simulator):
    target_states = []
    for i in range(2 ** qubits):
        qc = QuantumCircuit(qubits)
        binary_str = format(i, f'0{qubits}b')
        for j, bit in enumerate(binary_str):
            if bit == '1':
                qc.x(j)
        qc.append(QFT(num_qubits=qubits), list(range(qubits)))
        qc = transpile(qc, basis_gates=native_gates)
        qc.save_density_matrix()
        result = simulator.run(qc).result()
        target_states.append(result.data(0)['density_matrix'])
    return target_states

def evaluate_circuit_fitness(circuit, simulator, num_qubits, target_states):
    fitness = 0
    for i, target_state in enumerate(target_states):
        qc = QuantumCircuit(num_qubits)
        binary_str = format(i, f'0{num_qubits}b')
        for j, bit in enumerate(binary_str):
            if bit == '1':
                qc.x(j)
        qc.compose(transpile(circuit, basis_gates=native_gates), inplace=True)
        qc.save_density_matrix()
        result = simulator.run(qc).result()
        state = result.data(0)['density_matrix']
        fitness += state_fidelity(state, target_state)
    return fitness / len(target_states)

def evaluate_noisy_fitness_multiple_times(circuit, simulator, num_qubits, target_states, reps=1):
    return np.mean([evaluate_circuit_fitness(circuit, simulator, num_qubits, target_states) for _ in range(reps)])

# Data directories
num_qubits = 2
directories = [
    #f"Experiment Results/Optimiser_simple/{num_qubits} Qubit Simulation/Data",
    #f"Experiment Results/Optimiser_depth_reduction/{num_qubits} Qubit Simulation/Data",
    f"Experiment Results/Optimiser_noisy/{num_qubits} Qubit Simulation/Data",
    f"Experiment Results/Optimiser_noisy_depth_reduction/{num_qubits} Qubit Simulation/Data"
]

results_list = []
for directory in directories:
    print(directory)
    sim_type = os.path.basename(os.path.dirname(os.path.dirname(directory)))
    files = [f for f in os.listdir(directory) if f.endswith("_final_chromosomes.csv")]
    for file in files:
        df = pd.read_csv(os.path.join(directory, file))
        df['Chromosome'] = df['Chromosome'].apply(ast.literal_eval)
        chromosomes = df['Chromosome'].tolist()[:3]
        print(file)
        circuits = get_circuits(chromosomes)
        n_qubits = len(chromosomes[0][0])
        targets = get_qft_target_states(n_qubits, noiseless_simulator)

        noisy_fid = [evaluate_noisy_fitness_multiple_times(c, noisy_simulator, n_qubits, targets) for c in circuits]
        ideal_fid = [evaluate_circuit_fitness(c, noiseless_simulator, n_qubits, targets) for c in circuits]
        avg_drop = np.mean([(i - n) / i * 100 for i, n in zip(ideal_fid, noisy_fid)])

        results_list.append({
            "Optimiser Type": sim_type,
            "Run": file.split('_')[0],
            "Ideal Fidelity": np.mean(ideal_fid),
            "Noisy Fidelity": np.mean(noisy_fid),
            "Fidelity Drop (%)": avg_drop
        })

# Add textbook QFT baseline
qft = QFT(num_qubits)
targets = get_qft_target_states(num_qubits, noiseless_simulator)
ideal_qft = evaluate_circuit_fitness(qft, noiseless_simulator, num_qubits, targets)
noisy_qft = evaluate_noisy_fitness_multiple_times(qft, noisy_simulator, num_qubits, targets)
drop_qft = (ideal_qft - noisy_qft) / ideal_qft * 100

results_list.insert(0, {
    "Optimiser Type": "Traditional QFT",
    "Run": "N/A",
    "Ideal Fidelity": ideal_qft,
    "Noisy Fidelity": noisy_qft,
    "Fidelity Drop (%)": drop_qft
})

results_df = pd.DataFrame(results_list)

# Order optimisers
ordered_types = [
    "Traditional QFT", "Optimiser_simple",
    "Optimiser_depth_reduction", "Optimiser_noisy", "Optimiser_noisy_depth_reduction"
]
results_df["Optimiser Type"] = pd.Categorical(results_df["Optimiser Type"], categories=ordered_types, ordered=True)

# Compute final table
final_summary = []
for opt in ordered_types:
    group = results_df[results_df["Optimiser Type"] == opt]
    if not group.empty:
        avg_ideal = group["Ideal Fidelity"].mean()
        avg_noisy = group["Noisy Fidelity"].mean()
        perc_84 = np.percentile(group["Noisy Fidelity"], 84)
        drop = ((avg_ideal - avg_noisy) / avg_ideal) * 100

        final_summary.append({
            "Optimiser Type": opt,
            "Avg Ideal Fidelity": round(avg_ideal, 6),
            "Avg Noisy Fidelity": round(avg_noisy, 6),
            "% Drop in Avg Fidelity": round(drop, 6),
            "84th Percentile Fidelity": round(perc_84, 6)
        })

final_summary_df = pd.DataFrame(final_summary)
print(final_summary_df.to_string(index=False))
