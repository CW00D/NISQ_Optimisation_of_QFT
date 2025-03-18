#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Define the directories within the code
    directories = [
        "C:\\Users\\chris\\OneDrive\\Documents\\Uni\\Year 4\\Dissertation\\Project\\Experiment Results\\Optimiser_noisy\\2 Qubit Simulation"
    ]
    iteration = None  # Set to a specific iteration number if needed

    for directory in directories:
        fitness_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.startswith("run") and f.endswith("_fitness.csv")]
        all_fitness_values = []

        for file_path in fitness_files:
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue
            df = pd.read_csv(file_path)
            # If an iteration is specified, filter by it;
            # otherwise use the final iteration available in the file.
            if iteration is not None:
                df_iter = df[df["Iteration"] == iteration]
            else:
                final_iter = df["Iteration"].max()
                df_iter = df[df["Iteration"] == final_iter]
            # Extract the fitness column.
            fitness_values = df_iter["Fitness"].values
            all_fitness_values.extend(fitness_values)

        if not all_fitness_values:
            print(f"No valid data to plot for directory: {directory}")
            continue

        plt.figure(figsize=(10, 6))
        box = plt.boxplot(all_fitness_values, patch_artist=True, showmeans=False)
        title_iter = iteration if iteration is not None else final_iter
        n_runs = len(fitness_files)
        plt.title(f"Fitness Distribution (Average over {n_runs} runs)")
        plt.ylabel("Fitness")
        plt.ylim(0, 1)

        # Add horizontal lines for y-axis points
        y_ticks = plt.yticks()[0]
        for y in y_ticks:
            plt.axhline(y=y, color='gray', linestyle='--', linewidth=0.5)

        # Add labels for key points on the box charts
        for i, line in enumerate(box['medians']):
            x, y = line.get_xydata()[1]
            plt.text(x + 0.1, y, f'{y:.2f}', horizontalalignment='left', verticalalignment='bottom', fontsize=8, color='black', fontweight='bold')
        for i, line in enumerate(box['whiskers']):
            x, y = line.get_xydata()[1]
            plt.text(x + 0.1, y, f'{y:.2f}', horizontalalignment='left', verticalalignment='bottom', fontsize=8, color='black', fontweight='bold')

        # Save the box plot in the "Results" directory of the current folder
        output_dir = os.path.join(directory, "Results")
        os.makedirs(output_dir, exist_ok=True)
        output_name = os.path.basename(directory)
        output = os.path.join(output_dir, f"{output_name} Box Plot.png")

        plt.savefig(output, bbox_inches="tight")
        plt.close()
        print(f"Box plot saved to {output}")

if __name__ == "__main__":
    main()