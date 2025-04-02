
import optimiser_simple

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QFT
from qiskit.quantum_info import state_fidelity


qft_chromosome = [['z(0)', 'w'], ['h(0)', 'w'], ['-', 'cz(0,1)'], ['-', 'ryy(0,1,0.7845371797807829)'], ['w', 'h(1)'], ['cx(1,0)', '-'], ['sdg(0)', 'w'], ['t(0)', 'w'], ['swap(1,0)', '-']]
    
qft_circuit = optimiser_simple.get_circuits([qft_chromosome])


print("\n\n")
print(qft_circuit[0])
print("\n\n")




#Top Chromosome for Optimiser_simple:
#[['w', 'w'], ['w', 'h(1)'], ['t(0)', 'tdg(1)'], ['tdg(0)', 'w'], ['cy(1,0)', '-'], ['swap(1,0)', '-'], ['h(0)', 'tdg(1)'], ['-', 'cz(0,1)'], ['swap(1,0)', '-'], ['w', 'w']]
#--------------------------------------------------------------------------------
#Top Chromosome for Optimiser_depth_reduction:
#[['-', 'crx(0,1,1.570798744699224)'], ['cx(1,0)', '-'], ['-', 'z(1)'], ['h(0)', 'h(1)'], ['cx(1,0)', '-']]
#--------------------------------------------------------------------------------
#Top Chromosome for Optimiser_noisy:
#[['z(0)', 'w'], ['h(0)', 'w'], ['-', 'cz(0,1)'], ['-', 'ryy(0,1,0.7845371797807829)'], ['w', 'h(1)'], ['cx(1,0)', '-'], ['sdg(0)', 'w'], ['t(0)', 'w'], ['swap(1,0)', '-']]
#--------------------------------------------------------------------------------
#Top Chromosome for Optimiser_noisy_depth_reduction:
#[['-', 'rx(1,1.5753135223415262)'], ['cx(1,0)', '-'], ['-', 'h(1)'], ['rz(0,0.7939144640117518)', 'rx(1,0.7616351676962666)'], ['cz(1,0)', '-']]