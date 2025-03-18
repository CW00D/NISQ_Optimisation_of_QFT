import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime

def plot_averaged_results(simulator_type, simulation_name, run_numbers, output_dir):
    # Initialize lists to store the data
    all_ea_max = []
    all_ea_avg = []
    random_max = None
    random_avg = None

    # Read the iteration data from each run
    for run_number in run_numbers:
        iteration_data_filepath = os.path.join("Experiment Results", "Optimiser_" + simulator_type, simulation_name, f"run{run_number}_iteration_data.csv")
        if not os.path.exists(iteration_data_filepath):
            print(f"File not found: {iteration_data_filepath}")
            continue
        
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

    # Plot the results
    plt.clf()
    plt.plot(x_values, avg_ea_max, label="EA Max Fitness", color="blue")
    plt.plot(x_values, avg_ea_avg, label="EA Avg Fitness", color="blue", linestyle="--")
    plt.plot(x_values, random_max, label="Random Max Overall Fitness", color="orange")
    plt.plot(x_values, random_avg, label="Random Avg Fitness", color="orange", linestyle="--")
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2)
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.ylim(0, 1)
    plt.title(f"Fitness Over {len(avg_ea_max)} Iterations (Avg Over {len(run_numbers)} Runs)")
    plt.grid(True)
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    plt.savefig(os.path.join(output_dir, f"{simulation_name} Fitness Chart.png"), bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {os.path.join(output_dir, f'{simulation_name} Fitness Chart.png')}")

if __name__ == "__main__":
    simulation_name = "3 Qubit Simulation"  # Specify the simulation name
    simulator_type = "depth_reduction"
    run_numbers = [1, 2]  # Specify the run numbers
    output_dir = "Experiment Results/Optimiser_" + simulator_type + "/" + simulation_name + "/Results"
    plot_averaged_results(simulator_type, simulation_name, run_numbers, output_dir)