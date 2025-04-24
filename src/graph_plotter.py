import os
import pandas as pd
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import datetime
import numpy as np

def plot_averaged_results(simulator_type, simulation_name, output_dir):
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import datetime

    def moving_average(data, window_size):
        """Compute the moving average of a 1D array."""
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    all_ea_max = []
    all_ea_avg = []
    random_max = None
    random_avg = None

    iteration_data_dir = os.path.join("Experiment Results", "Optimiser_" + simulator_type, simulation_name, "Data")
    files = os.listdir(iteration_data_dir)
    iteration_data_files = [f for f in files if f.startswith("run") and f.endswith("_iteration_data.csv")]

    for iteration_data_file in iteration_data_files:
        df = pd.read_csv(os.path.join(iteration_data_dir, iteration_data_file))
        all_ea_max.append(df["EA Max Fitness"].values)
        all_ea_avg.append(df["EA Avg Fitness"].values)
        
        if random_max is None:
            random_max = df["Random Max Fitness"].values
            random_avg = df["Random Avg Fitness"].values

    all_ea_max = np.array(all_ea_max)
    all_ea_avg = np.array(all_ea_avg)

    avg_ea_max = np.mean(all_ea_max, axis=0)
    avg_ea_avg = np.mean(all_ea_avg, axis=0)
    std_ea_max = np.std(all_ea_max, axis=0)
    std_ea_avg = np.std(all_ea_avg, axis=0)

    # Apply a moving average to smooth the average population line
    window_size = 50  # Adjust this value to control the smoothing level
    smoothed_avg_ea_avg = moving_average(avg_ea_avg, window_size)
    smoothed_random_avg = moving_average(random_avg, window_size)

    x_values = np.arange(len(avg_ea_max))
    smoothed_x_values = np.arange(len(smoothed_avg_ea_avg)) + (window_size // 2)  # Adjust x-values for the moving average
    smoothed_random_x_values = np.arange(len(smoothed_random_avg)) + (window_size // 2)  # Adjust x-values for the random average
    step = len(x_values) // 10

    # Create mask for every n/10 iterations
    err_indices = np.arange(0, len(x_values), step)

    plt.figure(figsize=(10, 6))
    
    # EA plots
    plt.plot(x_values, avg_ea_max, label="EA Max Fitness", color="blue")
    plt.plot(smoothed_x_values, smoothed_avg_ea_avg, label="EA Avg Fitness (Smoothed)", color="blue", linestyle="--")
    plt.errorbar(err_indices, avg_ea_max[err_indices], yerr=std_ea_max[err_indices], fmt='o', color='blue', capsize=3)
    plt.errorbar(err_indices, avg_ea_avg[err_indices], yerr=std_ea_avg[err_indices], fmt='o', color='blue', capsize=3)

    # Random plots
    plt.plot(x_values, random_max, label="Random Max Fitness", color="orange")
    plt.plot(smoothed_random_x_values, smoothed_random_avg, label="Random Avg Fitness (Smoothed)", color="orange", linestyle="--")

    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2)
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.ylim(0, 1)
    plt.title(f"Fitness Over {len(x_values)} Iterations (Avg Over {len(iteration_data_files)} Runs)")
    plt.grid(True)

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    plt.savefig(os.path.join(output_dir, f"{simulation_name} Fitness Chart.png"), bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {os.path.join(output_dir, f'{simulation_name} Fitness Chart.png')}")

if __name__ == "__main__":
    simulation_name = "2 Qubit Simulation"  # Specify the simulation name
    
    simulator_type = "simple"
    #simulator_type = "depth_reduction"
    #simulator_type = "noisy"
    #simulator_type = "noisy_depth_reduction"
    
    output_dir = os.path.join("Experiment Results", "Optimiser_" + simulator_type, simulation_name, "Results")
    plot_averaged_results(simulator_type, simulation_name, output_dir)
