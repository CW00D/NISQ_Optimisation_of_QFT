# NISQ-QFT-MOEA

## Overview
This repository is part of my dissertation project, focused on developing a Multi-Objective Evolutionary Algorithm (MOEA) for optimizing the Quantum Fourier Transform (QFT) on Noisy Intermediate-Scale Quantum (NISQ) hardware. The project combines quantum computing principles with advanced optimization techniques to address challenges in noise resilience and gate fidelity.

## Goals
- Develop a hybrid representation scheme for quantum circuits combining serial and block-based approaches.
- Optimize QFT circuits for execution on NISQ hardware by balancing circuit fidelity and noise resilience.
- Benchmark the optimized circuits against traditional QFT implementations under realistic noise conditions.

## Repository Structure
```plaintext
NISQ_Optimisation_of_QFT/
├── docs/                  # Documentation files
├── src/                   # Source code for the project
├── experiments/           # Experiment scripts and data
├── tests/                 # Unit and integration tests
├── .github/               # GitHub-specific configurations
├── .gitignore             # Ignored files and folders
├── LICENSE                # Licensing information
└── README.md              # Project overview
```

## Prerequisites
- **Python**: Version 3.8 or higher is recommended.
- **Quantum Libraries**:
  - [Qiskit](https://qiskit.org/): For quantum circuit simulation and noise modeling.
  - [DEAP](https://deap.readthedocs.io/): For implementing the Multi-Objective Evolutionary Algorithm (MOEA).
- **Hardware**:
  - A system capable of running Python and handling computationally intensive tasks.
  - Access to a quantum simulator (e.g., Qiskit Aer or IBM Quantum Experience).

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<username>/NISQ-QFT-MOEA.git
   cd NISQ-QFT-MOEA
   ```

2. Set up a virtual environment:
   - **Linux/macOS**:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - **Windows**:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Explore the src/ directory for the MOEA implementation.
2. Run experiments using the scripts in the experiments/ folder.
3. View and analyze results in the notebooks/ directory.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
For any questions, feel free to reach out via christian.f.wood@gmail.com.




