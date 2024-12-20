import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import re

def extract_sensor_values_from_file(file_content: str):
    # Use regex to extract the list of sensors used
    sensor_list_match = re.search(r"sensors used: \[(.*?)\]", file_content)
    
    if sensor_list_match:
        # Extract sensor names and convert them to integers
        sensor_list_str = sensor_list_match.group(1)
        sensor_ids = re.findall(r'sensor_(\d+)', sensor_list_str)
        sensor_ids = [int(sensor_id) for sensor_id in sensor_ids]
        return sensor_ids
    else:
        return None

import os

def get_sensor_values_for_maps(base_folder: str):
    maps = []
    seeds = [1, 3, 4, 5, 7, 8, 9, 10, 12, 13]
    # Loop through each map folder (map0 to map9)
    for seed in seeds:
        map_folder = os.path.join(base_folder, f"seed{seed}")
        map_data = []
        
        # Loop through each of the files output10k.txt to output50k.txt
        for output_file in ["output10k.txt", "output15k.txt", "output20k.txt", "output25k.txt", "output30k.txt", "output35k.txt", "output40k.txt", "output45k.txt", "output50k.txt"]:
        # for output_file in ["output15k.txt", "output30k.txt", "output45k.txt"]:
            file_path = os.path.join(map_folder, output_file)
            
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    file_content = f.read()
                    sensor_ids = extract_sensor_values_from_file(file_content)
                    
                    # Replace the list with an empty list if it contains 25 elements
                    if sensor_ids and len(sensor_ids) == 25:
                        map_data.append([])
                    else:
                        map_data.append(sensor_ids)
            else:
                map_data.append([])  # If the file does not exist, add an empty list

        maps.append(map_data)
    
    return maps

def parse_results_file(file_path):
    # Initialize placeholders for each section
    yields_data, irrs_data, mses_data = [], [], []
    current_section = None
    row_indices_yields, row_indices_irrs, row_indices_mses = [], [], []
    columns_set = False  # Flag to track if columns are set dynamically

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Check which section we are in based on headers
            if "Yields (tonnes):" in line:
                current_section = 'yields'
                columns_set = False
                continue
            elif "Irrigation (cubic meters):" in line:
                current_section = 'irrigation'
                columns_set = False
                continue
            elif "Mean Squared Error (MSE):" in line:
                current_section = 'mse'
                columns_set = False
                continue
            
            # Skip empty lines and header lines with budget labels
            if line == "" or line.startswith("10k") or line.startswith("20k"):
                # Extract columns if not already set
                if not columns_set:
                    columns = line.split()[:]  # Exclude the row label
                    columns_set = True
                continue
            
            # Split line data and extract the row index
            row_data = line.split()
            try:
                row_index = int(row_data[0])  # Convert the first item to an integer row index
            except ValueError:
                continue  # Skip lines that don't start with an integer index
            row_values = row_data[1:]  # Remaining items are the data values
            
            # Add data to the correct section list and store row indices
            if current_section == 'yields':
                yields_data.append(row_values)
                row_indices_yields.append(row_index)
            elif current_section == 'irrigation':
                irrs_data.append(row_values)
                row_indices_irrs.append(row_index)
            elif current_section == 'mse':
                mses_data.append(row_values)
                row_indices_mses.append(row_index)
                
    # Convert lists to DataFrames, ensuring correct index and column labels
    yields_df = pd.DataFrame(yields_data, index=row_indices_yields, columns=columns).astype(float)
    irrs_df = pd.DataFrame(irrs_data, index=row_indices_irrs, columns=columns).astype(float)
    mses_df = pd.DataFrame(mses_data, index=row_indices_mses, columns=columns).astype(float)

    return yields_df, irrs_df, mses_df


# Parse each results.txt file
yields_soils, irrs_soils, mses_soils = parse_results_file("Energy Budget Experiments Soils/results.txt")
yields_soils_ig, irrs_soils_ig, mses_soils_ig = parse_results_file("Energy Budget Experiments Soils IG/results.txt")
yields_soils_greedy, irrs_soils_greedy, mses_soils_greedy = parse_results_file("Energy Budget Experiments Soils Greedy/results.txt")
# yields_soils_rseo, irrs_soils_rseo, mses_soils_rseo = parse_results_file("Energy Budget Experiments Soils RSEO/results.txt")
# for df in [yields_soils_rseo, irrs_soils_rseo, mses_soils_rseo]:
#     df.insert(0, '15k', 0)
#     df.insert(0, '10k', 0)
# Now yields_soils, irrs_soils, and mses_soils (and the IG equivalents) are DataFrames containing the data
print("Yields (Soils):\n", yields_soils)
print("Irrigation (Soils):\n", irrs_soils)
print("MSE (Soils):\n", mses_soils)
print("Yields (Soils IG):\n", yields_soils_ig)
print("Irrigation (Soils IG):\n", irrs_soils_ig)
print("MSE (Soils IG):\n", mses_soils_ig)
print("Yields (Soils Greedy):\n", yields_soils_greedy)
print("Irrigation (Soils Greedy):\n", irrs_soils_greedy)
print("MSE (Soils Greedy):\n", mses_soils_greedy)
# print("Yields (Soils RSEO):\n", yields_soils_rseo)
# print("Irrigation (Soils RSEO):\n", irrs_soils_rseo)
# print("MSE (Soils RSEO):\n", mses_soils_rseo)


# Calculate means and standard deviations for each metric
yields_mean_soils = yields_soils.mean(axis=0)
yields_std_soils = yields_soils.std(axis=0)
irrs_mean_soils = irrs_soils.mean(axis=0)
irrs_std_soils = irrs_soils.std(axis=0)
mses_mean_soils = mses_soils.mean(axis=0)
mses_std_soils = mses_soils.std(axis=0)

yields_mean_soils_ig = yields_soils_ig.mean(axis=0)
yields_std_soils_ig = yields_soils_ig.std(axis=0)
irrs_mean_soils_ig = irrs_soils_ig.mean(axis=0)
irrs_std_soils_ig = irrs_soils_ig.std(axis=0)
mses_mean_soils_ig = mses_soils_ig.mean(axis=0)
mses_std_soils_ig = mses_soils_ig.std(axis=0)

yields_mean_soils_greedy = yields_soils_greedy.mean(axis=0)
yields_std_soils_greedy = yields_soils_greedy.std(axis=0)
irrs_mean_soils_greedy = irrs_soils_greedy.mean(axis=0)
irrs_std_soils_greedy = irrs_soils_greedy.std(axis=0)
mses_mean_soils_greedy = mses_soils_greedy.mean(axis=0)
mses_std_soils_greedy = mses_soils_greedy.std(axis=0)

# yields_mean_soils_rseo = yields_soils_rseo.mean(axis=0)
# yields_std_soils_rseo = yields_soils_rseo.std(axis=0)
# irrs_mean_soils_rseo = irrs_soils_rseo.mean(axis=0)
# irrs_std_soils_rseo = irrs_soils_rseo.std(axis=0)
# mses_mean_soils_rseo = mses_soils_rseo.mean(axis=0)
# mses_std_soils_rseo = mses_soils_rseo.std(axis=0)

# X-axis labels and range for energy budgets
mse_x_labels = yields_soils.columns.to_list()
mse_x_range = np.arange(len(mse_x_labels))
ig_x_labels = yields_soils_ig.columns.to_list()
ig_x_range = np.arange(len(ig_x_labels))
greedy_x_labels = yields_soils_greedy.columns.to_list()
greedy_x_range = np.arange(len(greedy_x_labels))
# rseo_x_labels = yields_soils_rseo.columns.to_list()
# rseo_x_range = np.arange(len(rseo_x_labels))

# Create a figure with 4 subplots
# fig, axs = plt.subplots(4, 1, figsize=(12, 15), sharex=True)

# # Plotting Yields with error bars
# axs[0].errorbar(mse_x_range, yields_mean_soils, yerr=yields_std_soils, label='DRONE', marker='o', capsize=5)
# axs[0].errorbar(ig_x_range, yields_mean_soils_ig, yerr=yields_std_soils_ig, label='Fast-DRONE', marker='s', capsize=5)
# axs[0].errorbar(rseo_x_range, yields_mean_soils_rseo, yerr=yields_std_soils_rseo, label='RSEO', marker='d', capsize=5)
# axs[0].set_ylabel('Yield (tonnes)')
# axs[0].set_title('Total Yield vs Energy Budget')
# axs[0].grid(True)
# axs[0].legend()

# # Plotting Irrigation with error bars
# axs[1].errorbar(mse_x_range, irrs_mean_soils, yerr=irrs_std_soils, label='DRONE', marker='o', capsize=5)
# axs[1].errorbar(ig_x_range, irrs_mean_soils_ig, yerr=irrs_std_soils_ig, label='Fast-DRONE', marker='s', capsize=5)
# axs[1].errorbar(rseo_x_range, irrs_mean_soils_rseo, yerr=irrs_std_soils_rseo, label='RSEO', marker='d', capsize=5)
# axs[1].set_ylabel('Irrigation (cubic meters)')
# axs[1].set_title('Total Irrigation vs Energy Budget')
# axs[1].grid(True)
# axs[1].legend()

# # Plotting MSE with error bars
# axs[2].errorbar(mse_x_range, mses_mean_soils, yerr=mses_std_soils, label='DRONE', marker='o', capsize=5)
# axs[2].errorbar(ig_x_range, mses_mean_soils_ig, yerr=mses_std_soils_ig, label='Fast-DRONE', marker='s', capsize=5)
# axs[2].errorbar(rseo_x_range, mses_mean_soils_rseo, yerr=mses_std_soils_rseo, label='RSEO', marker='d', capsize=5)
# axs[2].set_ylabel('MSE')
# axs[2].set_xlabel('Energy Budget')
# axs[2].set_title('Mean Squared Error (MSE) vs Energy Budget')
# axs[2].grid(True)
# axs[2].legend()


maps_drone = get_sensor_values_for_maps("Energy Budget Experiments Soils")
maps_fast_drone = get_sensor_values_for_maps("Energy Budget Experiments Soils IG")
maps_greedy = get_sensor_values_for_maps("Energy Budget Experiments Soils Greedy")
# maps_rseo = get_sensor_values_for_maps("Energy Budget Experiments Soils RSEO")

def get_sensor_counts(maps):
    # Calculate the average number of sensors per position, counting empty lists as 25
    average_num_sensors = []
    sensors_std = []

    # Loop over each index position across sublists
    for idx in range(9):
        num_sensors = []
        for map_data in maps:
            # Check if the current map has a sublist at the position idx
            if idx < len(map_data):
                # Count items in sublist, using 25 if it is empty
                num_sensors.append(len(map_data[idx]) if map_data[idx] else 25)
            else:
                # If sublist does not exist in this map, assume it's empty (count as 25)
                num_sensors.append(25)
        # Calculate mean and standard deviation for this index position
        average_num_sensors.append(np.mean(num_sensors))
        sensors_std.append(np.std(num_sensors))
    return average_num_sensors, sensors_std

drone_avg_sensors, drone_sensors_std = get_sensor_counts(maps_drone)
fast_drone_avg_sensors, fast_drone_sensors_std = get_sensor_counts(maps_fast_drone)
greedy_avg_sensors, greedy_sensors_std = get_sensor_counts(maps_greedy)
# rseo_avg_sensors, rseo_sensors_std = get_sensor_counts(maps_rseo)
# rseo_avg_sensors[0], rseo_sensors_std[0] = 0, 0
# rseo_avg_sensors[1], rseo_sensors_std[1] = 0, 0

# # Plot the average number of sensors with error bars on axs[3]
# axs[3].errorbar(mse_x_range, drone_avg_sensors, yerr=drone_sensors_std, label='DRONE', marker='o', capsize=5)
# axs[3].errorbar(ig_x_range, fast_drone_avg_sensors, yerr=fast_drone_sensors_std, label='Fast-DRONE', marker='s', capsize=5)
# axs[3].errorbar(mse_x_range, rseo_avg_sensors, yerr=rseo_sensors_std, label='RSEO', marker='d', capsize=5)
# axs[3].set_ylabel('Average Number of Sensors')
# axs[3].set_xlabel('Energy Budget')
# axs[3].set_title('Average Number of Sensors Selected for Each Energy Budget')
# axs[3].grid(True)

# # Customize x-axis labels
# axs[3].set_xticks(mse_x_range)
# axs[3].set_xticklabels(mse_x_labels)

# # Adjust layout and display the plots
# plt.tight_layout()
# plt.show()




# Plot everything
 
# # X-axis labels and range for energy budgets
# x_labels = mse_x_labels  # Use common labels from one dataset, e.g., mse_x_labels
# x = np.arange(len(x_labels))  # Generate positions for the x-axis

# Width of each bar
bar_width = 0.25

# # Create a figure with 4 subplots
# fig, axs = plt.subplots(4, 1, figsize=(12, 20), sharex=True)

# # Plotting Yields with error bars
# # axs[0].bar(x - bar_width, yields_mean_soils, width=bar_width, yerr=yields_std_soils, capsize=5, label='DRONE')
# # axs[0].bar(x, yields_mean_soils_ig, width=bar_width, yerr=yields_std_soils_ig, capsize=5, label='Fast-DRONE')
# # axs[0].bar(x + bar_width, yields_mean_soils_greedy, width=bar_width, yerr=yields_std_soils_greedy, capsize=5, label='Greedy')
# axs[0].bar(x - bar_width, yields_mean_soils, width=bar_width, capsize=5, label='DRONE')
# axs[0].bar(x, yields_mean_soils_ig, width=bar_width, capsize=5, label='Fast-DRONE')
# axs[0].bar(x + bar_width, yields_mean_soils_greedy, width=bar_width, capsize=5, label='Greedy')
# axs[0].set_ylim(1375, 1385)
# # axs[0].bar(x + bar_width, yields_mean_soils_rseo, width=bar_width, yerr=yields_std_soils_rseo, capsize=5, label='RSEO')
# axs[0].set_ylabel('Yield (tonnes)')
# axs[0].set_title('Total Yield vs Energy Budget')
# axs[0].legend()
# axs[0].grid(True)

# # Plotting Irrigation with error bars
# # axs[1].bar(x - bar_width, irrs_mean_soils, width=bar_width, yerr=irrs_std_soils, capsize=5, label='DRONE')
# # axs[1].bar(x, irrs_mean_soils_ig, width=bar_width, yerr=irrs_std_soils_ig, capsize=5, label='Fast-DRONE')
# # axs[1].bar(x + bar_width, irrs_mean_soils_greedy, width=bar_width, yerr=irrs_std_soils_greedy, capsize=5, label='Greedy')
# axs[1].bar(x - bar_width, irrs_mean_soils, width=bar_width, capsize=5, label='DRONE')
# axs[1].bar(x, irrs_mean_soils_ig, width=bar_width, capsize=5, label='Fast-DRONE')
# axs[1].bar(x + bar_width, irrs_mean_soils_greedy, width=bar_width, capsize=5, label='Greedy')
# axs[1].set_ylim(325000, 350000)
# # axs[1].bar(x + bar_width, irrs_mean_soils_rseo, width=bar_width, yerr=irrs_std_soils_rseo, capsize=5, label='RSEO')
# axs[1].set_ylabel('Irrigation (cubic meters)')
# axs[1].set_title('Total Irrigation vs Energy Budget')
# axs[1].legend()
# axs[1].grid(True)

# # Plotting MSE with error bars
# # axs[2].bar(x - bar_width, mses_mean_soils, width=bar_width, yerr=mses_std_soils, capsize=5, label='DRONE')
# # axs[2].bar(x, mses_mean_soils_ig, width=bar_width, yerr=mses_std_soils_ig, capsize=5, label='Fast-DRONE')
# # axs[2].bar(x + bar_width, mses_mean_soils_greedy, width=bar_width, yerr=mses_std_soils_greedy, capsize=5, label='Greedy')
# axs[2].bar(x - bar_width, mses_mean_soils, width=bar_width, capsize=5, label='DRONE')
# axs[2].bar(x, mses_mean_soils_ig, width=bar_width, capsize=5, label='Fast-DRONE')
# axs[2].bar(x + bar_width, mses_mean_soils_greedy, width=bar_width, capsize=5, label='Greedy')
# # axs[2].bar(x + bar_width, mses_mean_soils_rseo, width=bar_width, yerr=mses_std_soils_rseo, capsize=5, label='RSEO')
# axs[2].set_ylabel('MSE')
# # axs[2].set_xlabel('Energy Budget')
# axs[2].set_title('Mean Squared Error (MSE) vs Energy Budget')
# axs[2].legend()
# axs[2].grid(True)

# # Plotting Average Number of Sensors with error bars
# # axs[3].bar(x - bar_width, drone_avg_sensors, width=bar_width, yerr=drone_sensors_std, capsize=5, label='DRONE')
# # axs[3].bar(x, fast_drone_avg_sensors, width=bar_width, yerr=fast_drone_sensors_std, capsize=5, label='Fast-DRONE')
# # axs[3].bar(x + bar_width, greedy_avg_sensors, width=bar_width, yerr=greedy_sensors_std, capsize=5, label='Greedy')
# axs[3].bar(x - bar_width, drone_avg_sensors, width=bar_width, capsize=5, label='DRONE')
# axs[3].bar(x, fast_drone_avg_sensors, width=bar_width, capsize=5, label='Fast-DRONE')
# axs[3].bar(x + bar_width, greedy_avg_sensors, width=bar_width, capsize=5, label='Greedy')
# # axs[3].bar(x + bar_width, rseo_avg_sensors, width=bar_width, yerr=rseo_sensors_std, capsize=5, label='RSEO')
# axs[3].set_ylabel('Average Number of Sensors')
# axs[3].set_xlabel('Energy Budget')
# axs[3].set_title('Average Number of Sensors Selected for Each Energy Budget')
# axs[3].legend()
# axs[3].grid(True)

# # Customize x-axis labels for all subplots
# plt.xticks(ticks=x, labels=x_labels)

# # Adjust layout to prevent overlap
# plt.tight_layout()
# plt.show()


# Assuming mse_x_labels, yields_mean_soils, irrs_mean_soils, etc. are already defined
# Filter the indices for '15k', '30k', '45k', and '50k'
selected_labels = ['15k', '30k', '45k']
selected_indices = [i for i, label in enumerate(mse_x_labels) if label in selected_labels]
non_predictive_index = [i for i, label in enumerate(mse_x_labels) if label == '50k']

# Convert selected_indices to a numpy array to ensure compatibility
selected_indices = np.array(selected_indices)

# Create a figure with 4 subplots
fig, axs = plt.subplots(4, 1, figsize=(12, 20), sharex=True)

# Plotting Yields with error bars (select values corresponding to selected indices)
axs[0].bar(selected_indices/2 - bar_width, np.array(yields_mean_soils)[selected_indices], width=bar_width, capsize=5, label='DRONE', color='C2')
axs[0].bar(selected_indices/2, np.array(yields_mean_soils_ig)[selected_indices], width=bar_width, capsize=5, label='Fast-DRONE', color='C1')
axs[0].bar(selected_indices/2 + bar_width, np.array(yields_mean_soils_greedy)[selected_indices], width=bar_width, capsize=5, label='Greedy', color='C3')
axs[0].axhline(y=np.array(yields_mean_soils)[non_predictive_index], color='r', linestyle='--', label='No Predictions')  # Horizontal line for '50k'
axs[0].set_ylim(1375, 1385)
axs[0].set_ylabel('Yield (tonnes)')
axs[0].set_title('Total Yield vs Energy Budget')
axs[0].legend()
axs[0].grid(True)

# Plotting Irrigation with error bars (select values corresponding to selected indices)
axs[1].bar(selected_indices/2 - bar_width, np.array(irrs_mean_soils)[selected_indices], width=bar_width, capsize=5, label='DRONE', color='C2')
axs[1].bar(selected_indices/2, np.array(irrs_mean_soils_ig)[selected_indices], width=bar_width, capsize=5, label='Fast-DRONE', color='C1')
axs[1].bar(selected_indices/2 + bar_width, np.array(irrs_mean_soils_greedy)[selected_indices], width=bar_width, capsize=5, label='Greedy', color='C3')
axs[1].axhline(y=np.array(irrs_mean_soils)[non_predictive_index], color='r', linestyle='--', label='No Predictions')  # Horizontal line for '50k'
axs[1].set_ylim(325000, 350000)
axs[1].set_ylabel('Irrigation (cubic meters)')
axs[1].set_title('Total Irrigation vs Energy Budget')
axs[1].legend()
axs[1].grid(True)

# Plotting MSE with error bars (select values corresponding to selected indices)
axs[2].bar(selected_indices/2 - bar_width, np.array(mses_mean_soils)[selected_indices], width=bar_width, capsize=5, label='DRONE', color='C2')
axs[2].bar(selected_indices/2, np.array(mses_mean_soils_ig)[selected_indices], width=bar_width, capsize=5, label='Fast-DRONE', color='C1')
axs[2].bar(selected_indices/2 + bar_width, np.array(mses_mean_soils_greedy)[selected_indices], width=bar_width, capsize=5, label='Greedy', color='C3')
# axs[2].axhline(y=np.array(mses_mean_soils)[non_predictive_index], color='r', linestyle='--', label='50k')  # Horizontal line for '50k'
axs[2].set_ylabel('MSE')
axs[2].set_title('Mean Squared Error (MSE) vs Energy Budget')
axs[2].legend()
axs[2].grid(True)

# Plotting Average Number of Sensors with error bars (select values corresponding to selected indices)
axs[3].bar(selected_indices/2 - bar_width, np.array(drone_avg_sensors)[selected_indices], width=bar_width, capsize=5, label='DRONE', color='C2')
axs[3].bar(selected_indices/2, np.array(fast_drone_avg_sensors)[selected_indices], width=bar_width, capsize=5, label='Fast-DRONE', color='C1')
axs[3].bar(selected_indices/2 + bar_width, np.array(greedy_avg_sensors)[selected_indices], width=bar_width, capsize=5, label='Greedy', color='C3')
axs[3].axhline(y=np.array(drone_avg_sensors)[non_predictive_index], color='r', linestyle='--', label='Maximum')  # Horizontal line for '50k'
axs[3].set_ylabel('Average Number of Sensors')
axs[3].set_xlabel('Energy Budget')
axs[3].set_title('Average Number of Sensors Selected for Each Energy Budget')
axs[3].legend()
axs[3].grid(True)

# Customize x-axis labels for all subplots
plt.xticks(ticks=selected_indices/2, labels=[mse_x_labels[i] for i in selected_indices])

# Adjust layout to prevent overlap
plt.tight_layout()
plt.savefig('Aquacrop Budget Final.png')
plt.show()