import os
import ast
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid thread issues
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT
from qiskit.quantum_info import state_fidelity
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error, phase_damping_error
from optimiser_noisy import get_circuits

# -----------------------------
# Noise Model Configuration
# -----------------------------
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
    "ccx": 0.1, "cswap": 0.12
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

noisy_simulator = AerSimulator(method='density_matrix', noise_model=noise_model)
noiseless_simulator = AerSimulator(method='density_matrix')
native_gates = noiseless_simulator.configuration().basis_gates

# -----------------------------
# Evaluation Functions
# -----------------------------
def get_qft_target_states(num_qubits, simulator):
    target_states = []
    for i in range(2 ** num_qubits):
        qc = QuantumCircuit(num_qubits)
        binary = format(i, f'0{num_qubits}b')
        for j, bit in enumerate(binary):
            if bit == '1':
                qc.x(j)
        qc.append(QFT(num_qubits), range(num_qubits))
        qc = transpile(qc, basis_gates=native_gates)
        qc.save_density_matrix()
        result = simulator.run(qc).result()
        target_states.append(result.data(0)['density_matrix'])
    return target_states

def evaluate_circuit_fidelity(circuit, simulator, num_qubits, target_states):
    total_fid = 0
    for i, target_state in enumerate(target_states):
        qc = QuantumCircuit(num_qubits)
        binary = format(i, f'0{num_qubits}b')
        for j, bit in enumerate(binary):
            if bit == '1':
                qc.x(j)
        transpiled = transpile(circuit, basis_gates=native_gates)
        qc.compose(transpiled, inplace=True)
        qc.save_density_matrix()
        result = simulator.run(qc).result()
        fid = state_fidelity(result.data(0)['density_matrix'], target_state)
        total_fid += fid
    return total_fid / len(target_states)

# -----------------------------
# Box Plot Generator
# -----------------------------
def generate_fidelity_boxplot(data_dir, num_qubits):
    run_files = [f for f in os.listdir(data_dir) if f.endswith("_final_chromosomes.csv")]
    if not run_files:
        print(f"No CSV files found in {data_dir}")
        return

    all_noisy_fid = []
    all_noiseless_fid = []

    for f in run_files:
        df = pd.read_csv(os.path.join(data_dir, f))
        df['Chromosome'] = df['Chromosome'].apply(ast.literal_eval)
        chromosomes = df['Chromosome'].tolist()[:3]
        circuits = get_circuits(chromosomes)
        target_states = get_qft_target_states(num_qubits, noiseless_simulator)

        for circ in circuits:
            noiseless_fid = evaluate_circuit_fidelity(circ, noiseless_simulator, num_qubits, target_states)
            noisy_fid = evaluate_circuit_fidelity(circ, noisy_simulator, num_qubits, target_states)
            all_noiseless_fid.append(noiseless_fid)
            all_noisy_fid.append(noisy_fid)

    if not all_noiseless_fid or not all_noisy_fid:
        print(f"No valid fidelity data found in {data_dir}")
        return

    # Plot
    plt.figure(figsize=(8, 6))
    box = plt.boxplot([all_noiseless_fid, all_noisy_fid],
                      patch_artist=True,
                      tick_labels=["Noiseless", "Noisy"],
                      showmeans=True)

    colors = ['#66c2a5', '#fc8d62']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    num_runs = len(run_files)
    plt.title(f"Best Circuit Box Plots (Over {num_runs} Iterations)")
    plt.ylabel("Fidelity")
    plt.ylim(0.8, 1.01)
    plt.grid(True, linestyle='--', alpha=0.5)

    output_dir = os.path.join(os.path.dirname(data_dir), "Results")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "Box Plots.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved boxplot to: {out_path}")

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    num_qubits = 3  # Change to 3 for the 3-qubit case

    dirs = [
        f"Experiment Results/Optimiser_simple/{num_qubits} Qubit Simulation/Data",
        f"Experiment Results/Optimiser_depth_reduction/{num_qubits} Qubit Simulation/Data",
        f"Experiment Results/Optimiser_noisy/{num_qubits} Qubit Simulation/Data",
        f"Experiment Results/Optimiser_noisy_depth_reduction/{num_qubits} Qubit Simulation/Data"
    ]

    for d in dirs:
        generate_fidelity_boxplot(d, num_qubits)
