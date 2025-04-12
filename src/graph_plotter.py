import os
import pandas as pd
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import datetime
import numpy as np

def plot_averaged_results(simulator_type, simulation_name, output_dir):
    # Initialize lists to store the data
    all_ea_max = []
    all_ea_avg = []
    random_max = None
    random_avg = None

    # Directory containing the iteration data files
    iteration_data_dir = os.path.join("Experiment Results", "Optimiser_" + simulator_type, simulation_name, "Data")

    # List all files in the directory
    files = os.listdir(iteration_data_dir)

    # Filter out the relevant CSV files
    iteration_data_files = [f for f in files if f.startswith("run") and f.endswith("_iteration_data.csv")]

    # Read the iteration data from each file
    for iteration_data_file in iteration_data_files:
        iteration_data_filepath = os.path.join(iteration_data_dir, iteration_data_file)
        df = pd.read_csv(iteration_data_filepath)
        all_ea_max.append(df["EA Max Fitness"].values)
        all_ea_avg.append(df["EA Avg Fitness"].values)
        
        if random_max is None:
            random_max = df["Random Max Fitness"].values
            random_avg = df["Random Avg Fitness"].values

    # Compute the averages
    avg_ea_max = [sum(run[i] for run in all_ea_max) / len(all_ea_max) for i in range(len(all_ea_max[0]))]
    avg_ea_avg = [sum(run[i] for run in all_ea_avg) / len(all_ea_avg) for i in range(len(all_ea_avg[0]))]
    x_values = list(range(len(avg_ea_max)))

    # Generate distinct colors for individual runs (excluding blue and orange)
    np.random.seed(42)
    colors = [plt.cm.tab10(i) for i in range(10) if i not in [0, 1]]  # Exclude blue (0) and orange (1)
    
    # Plot the results
    plt.clf()
    
    # Plot individual runs with distinct colors and slightly more intense transparency
    for i, run in enumerate(all_ea_max):
        plt.plot(x_values, run, color=colors[i % len(colors)], alpha=0.5)
    for i, run in enumerate(all_ea_avg):
        plt.plot(x_values, run, color=colors[i % len(colors)], linestyle="--", alpha=0.5)
    
    # Plot averaged results
    plt.plot(x_values, avg_ea_max, label="EA Max Fitness", color="blue")
    plt.plot(x_values, avg_ea_avg, label="EA Avg Fitness", color="blue", linestyle="--")
    plt.plot(x_values, random_max, label="Random Max Overall Fitness", color="orange")
    plt.plot(x_values, random_avg, label="Random Avg Fitness", color="orange", linestyle="--")
    
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2)
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.ylim(0, 1)
    plt.title(f"Fitness Over {len(avg_ea_max)} Iterations (Avg Over {len(iteration_data_files)} Runs)")
    plt.grid(True)
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    plt.savefig(os.path.join(output_dir, f"{simulation_name} Fitness Chart.png"), bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {os.path.join(output_dir, f'{simulation_name} Fitness Chart.png')}")

if __name__ == "__main__":
    simulation_name = "3 Qubit Simulation"  # Specify the simulation name
    
    #simulator_type = "simple"
    #simulator_type = "depth_reduction"
    #simulator_type = "noisy"
    simulator_type = "noisy_depth_reduction"
    
    output_dir = os.path.join("Experiment Results", "Optimiser_" + simulator_type, simulation_name, "Results")
    plot_averaged_results(simulator_type, simulation_name, output_dir)
