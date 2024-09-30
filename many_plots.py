import os
os.environ['DEVELOPMENT'] = 'DEVELOPMENT'  # Run purely in python

# suppress warning messages
import logging
import warnings
logging.getLogger().setLevel(logging.ERROR)
warnings.simplefilter(action='ignore', category=FutureWarning)

from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, FieldMngt, GroundWater, IrrigationManagement
from aquacrop.utils import prepare_weather, get_filepath

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

# def run_simulation(moisture_threshold, field_dimension=1):

#     # Prepare weather data
#     path = get_filepath('champion_climate.txt')
#     wdf = prepare_weather(path)

#     sim_start = '1982/05/01'
#     sim_end = '1983/05/01'
#     soil = Soil('SandyLoam')
#     SAT = soil.profile['th_s'][0]  # For when moisture_threshold is a percentage
#     crop = Crop('Maize', planting_date='05/01')
#     # initWC = InitialWaterContent(value=['FC'])
    

#     # Create a results list to store the results of each simulation
#     all_results = []

#     # Iterate over each grid cell in the field
#     for row in range(field_dimension):
#         for col in range(field_dimension):
#             print(f"Running simulation for grid cell ({row+1}, {col+1})")
#             initWC_value = random.random() * 100
#             initWC = InitialWaterContent(wc_type='Pct', value=[initWC_value])
#             print(f'random initial water content: {initWC_value}%')
#             # Initialize an empty schedule for this grid cell
#             schedule = pd.DataFrame(columns=['Date', 'Depth'])

#             # Set the simulation end date for weekly simulation
#             current_end = pd.to_datetime(sim_start) + pd.Timedelta(days=7)

#             # Run the simulation week by week
#             while current_end <= pd.to_datetime(sim_end):
#                 # Create model with the current schedule
#                 scheduled = IrrigationManagement(irrigation_method=3, Schedule=schedule)
#                 model = AquaCropModel(sim_start,
#                                       current_end.strftime('%Y/%m/%d'),
#                                       wdf,
#                                       soil,
#                                       crop,
#                                       initial_water_content=initWC,
#                                       irrigation_management=scheduled)
                
#                 # Run the model
#                 model.run_model(till_termination=True)

#                 # Get the water storage output
#                 water_storage = model._outputs.water_storage

#                 # Traverse the dataframe backwards to get the last day's moisture value
#                 for i in range(len(water_storage) - 1, -1, -1):
#                     if water_storage['growing_season'].iloc[i] == 1:
#                         current_moisture = water_storage['th1'].iloc[i]
#                         break

#                 # Calculate the current moisture percentage
#                 current_moisture_percentage = (current_moisture / SAT) * 100

#                 # Decide on the irrigation management for the next week
#                 next_week_dates = pd.date_range(current_end + pd.Timedelta(days=1), 
#                                                 current_end + pd.Timedelta(days=7))
#                 next_week_depths = [0] * len(next_week_dates)

#                 # When the moisture threshold is specified as a percentage
#                 if current_moisture_percentage < moisture_threshold:
#                     next_week_depths[0] = 25
#                 new_entries = pd.DataFrame({'Date': next_week_dates.date, 'Depth': next_week_depths})

#                 # Update the irrigation schedule
#                 schedule = pd.concat([schedule, new_entries])

#                 # Move to the next week
#                 current_end += pd.Timedelta(days=7)

#             # Final run for the entire season
#             final_scheduled = IrrigationManagement(irrigation_method=3, Schedule=schedule)
#             model = AquaCropModel(sim_start,
#                                   sim_end,
#                                   wdf,
#                                   soil,
#                                   crop,
#                                   initial_water_content=initWC,
#                                   irrigation_management=final_scheduled)
#             model.run_model(till_termination=True)

#             # Get the final stats
#             final_stats = model._outputs.final_stats
#             water_flux = model._outputs.water_flux
#             water_storage = model._outputs.water_storage

#             # Append the results for this grid cell
#             all_results.append({
#                 'grid_cell': (row+1, col+1),
#                 'final_stats': final_stats,
#                 'water_flux': water_flux,
#                 'water_storage': water_storage
#             })

#     return all_results

def run_simulation(moisture_threshold, num_fields=1):
    # Prepare weather data
    path = get_filepath('champion_climate.txt')
    wdf = prepare_weather(path)
    initWC = InitialWaterContent(value=['FC'])
    sim_start = '1982/05/01'
    sim_end = '1983/05/01'
    crop = Crop('Maize', planting_date='05/01')

    # Initialize an empty list to hold simulation data for each grid cell
    all_results = []
    
    # Initialize schedule and end date for each grid cell
    simulations = []
    
    soil_values = []
    for first in range(10, 91, 10):
        for second in range(10, 91-first, 10):
            soil_values.append((first, second, 100 - first - second))
    # random.shuffle(soil_values)
    soil_values = soil_values[:num_fields]   # Get a random soil value for each field
    # Prepare simulation for each grid cell in the field
    counter = 0
    for field in range(num_fields):
        soil_value = soil_values[counter]
        print(f"Setting up simulation for field {field} with soil value {soil_value}")
        print(f'Sand: {soil_value[0]}, Clay: {soil_value[1]}, Silt: {soil_value[2]}')
        soil = Soil('custom')
        soil.add_layer_from_texture(thickness=soil.zSoil,
                                Sand=soil_value[0],Clay=soil_value[1],
                                OrgMat=2.5,penetrability=100)
        SAT = soil.profile['th_s'][0]  # For when moisture_threshold is a percentage
        # Create initial schedule and current end date for weekly simulation
        schedule = pd.DataFrame(columns=['Date', 'Depth'])
        current_end = pd.to_datetime(sim_start) + pd.Timedelta(days=7)
        
        # Store simulation state for this grid cell
        simulations.append({
            'field': soil_value,
            'schedule': schedule,
            'soil': soil,
            'current_end': current_end,
            'current_moisture_percentage': None,
            'schedule_updated': False
        })
        counter += 1

    # Simulate week by week across all grid cells simultaneously
    season_complete = False
    while not season_complete:
        season_complete = True  # Assume all simulations are complete until proven otherwise

        # Run each simulation for the next week
        for sim in simulations:
            current_end = sim['current_end']
            
            if current_end <= pd.to_datetime(sim_end):
                season_complete = False  # At least one simulation still has weeks left

                # Create and run model for this grid cell for the current week
                scheduled = IrrigationManagement(irrigation_method=3, Schedule=sim['schedule'])
                model = AquaCropModel(sim_start,
                                      current_end.strftime('%Y/%m/%d'),
                                      wdf,
                                      soil=sim['soil'],
                                      crop=crop,
                                      initial_water_content=initWC,
                                      irrigation_management=scheduled)
                
                model.run_model(till_termination=True)
                
                # Get the water storage output
                water_storage = model._outputs.water_storage
                
                # Traverse the dataframe backwards to get the last day's moisture value
                for i in range(len(water_storage) - 1, -1, -1):
                    if water_storage['growing_season'].iloc[i] == 1:
                        current_moisture = water_storage['th1'].iloc[i]
                        break
                
                # Calculate the current moisture percentage
                current_moisture_percentage = (current_moisture / SAT) * 100
                sim['current_moisture_percentage'] = current_moisture_percentage

                # Decide on irrigation for the next week
                next_week_dates = pd.date_range(current_end + pd.Timedelta(days=1), 
                                                current_end + pd.Timedelta(days=7))
                next_week_depths = [0] * len(next_week_dates)
                
                print(f'On the week ending on {current_end}, plot {sim["field"]} has moisture {current_moisture_percentage}')
                if current_moisture_percentage < moisture_threshold:
                    print(f'This is below the {moisture_threshold}% Threshold, so we will irrigate')
                    next_week_depths[0] = 25

                new_entries = pd.DataFrame({'Date': next_week_dates.date, 'Depth': next_week_depths})
                sim['schedule'] = pd.concat([sim['schedule'], new_entries])
                
                # Update the current end date to the next week
                sim['current_end'] += pd.Timedelta(days=7)
        
    # Finalize all simulations for the full season
    for sim in simulations:
        final_scheduled = IrrigationManagement(irrigation_method=3, Schedule=sim['schedule'])
        model = AquaCropModel(sim_start,
                              sim_end,
                              wdf,
                              soil=sim['soil'],
                              crop=crop,
                              initial_water_content=initWC,
                              irrigation_management=final_scheduled)
        model.run_model(till_termination=True)

        # Get the final stats and water flux for each grid cell
        final_stats = model._outputs.final_stats
        water_flux = model._outputs.water_flux
        water_storage = model._outputs.water_storage

        # Append the results for this grid cell
        all_results.append({
            'field': sim['field'],
            'final_stats': final_stats,
            'water_flux': water_flux,
            'water_storage': water_storage
        })

    return all_results

results = run_simulation(moisture_threshold=50, num_fields=1)

def plot_daily_irrigation(results):
    plt.figure(figsize=(12, 6))

    for result in results:
        #Plot the irrigation of each individual day
        irrigation_data = result['water_flux'][['time_step_counter', 'IrrDay']]
        mask = (irrigation_data['time_step_counter'] == 0) & (irrigation_data['time_step_counter'].index != 0)

        # Find the index of the first occurrence of this condition
        first_zero_index = irrigation_data[mask].index.min()

        # Drop all rows from this index onwards if such an index exists
        if pd.notna(first_zero_index):
            irrigation_data = irrigation_data.iloc[:first_zero_index]
        print(irrigation_data)

        print(irrigation_data.columns)
        # Plot the daily irrigation amounts
        plt.plot(irrigation_data['time_step_counter'].to_numpy(), irrigation_data['IrrDay'].to_numpy(), marker='o', label=f'Soil {result["field"]}')

    plt.xlabel('Day')
    plt.ylabel('Irrigation (mm)')
    plt.title('Daily Irrigation Amounts for All Fields')
    plt.grid(True)
    plt.legend(title='Soil Values')
    plt.show()
    print(results)

# def plot_weekly_irrigation(results):
#     plt.figure(figsize=(12, 6))

#     for result in results:
#         #Plot the irrigation of each individual day
#         irrigation_data = result['water_flux'][['time_step_counter', 'IrrDay']]
#         irrigation_data = irrigation_data[irrigation_data['time_step_counter'] % 7 == 1]
#         mask = (irrigation_data['time_step_counter'] == 0) & (irrigation_data['time_step_counter'].index != 0)

#         # Find the index of the first occurrence of this condition
#         first_zero_index = irrigation_data[mask].index.min()

#         # Drop all rows from this index onwards if such an index exists
#         if pd.notna(first_zero_index):
#             irrigation_data = irrigation_data.iloc[:first_zero_index]
#         print(irrigation_data)

#         print(irrigation_data.columns)
#         # Plot the daily irrigation amounts
#         plt.plot(irrigation_data['time_step_counter'].to_numpy(), irrigation_data['IrrDay'].to_numpy(), marker='o', label=f'Soil {result["field"]}')

#     plt.xlabel('Day')
#     plt.ylabel('Irrigation (mm)')
#     plt.title('Weekly Irrigation Amounts for All Fields')
#     plt.grid(True)
#     plt.legend(title='Soil Values')
#     plt.show()
#     print(results)

def plot_weekly_irrigation(results):
    plt.figure(figsize=(12, 6))

    # Number of soil types
    num_soils = len(results)

    # Create bar width and the positions on the x-axis
    bar_width = 1 / (num_soils + 1)  # To avoid overlap, make sure bars are smaller
    offset = np.arange(len(results[0]['water_flux'][results[0]['water_flux']['time_step_counter'] % 7 == 1]))

    for i, result in enumerate(results):
        # Filter irrigation data for weekly intervals
        irrigation_data = result['water_flux'][['time_step_counter', 'IrrDay']]
        irrigation_data = irrigation_data[irrigation_data['time_step_counter'] % 7 == 1]
        mask = (irrigation_data['time_step_counter'] == 0) & (irrigation_data['time_step_counter'].index != 0)

        # Find the index of the first occurrence of this condition
        first_zero_index = irrigation_data[mask].index.min()

        # Drop all rows from this index onwards if such an index exists
        if pd.notna(first_zero_index):
            irrigation_data = irrigation_data.iloc[:first_zero_index]

        # Plot the weekly irrigation amounts as a bar graph
        x_positions = np.arange(len(irrigation_data)) + i * 0.1  # Add a small offset for each soil to prevent overlap
        plt.bar(x_positions, irrigation_data['IrrDay'].to_numpy(), width=bar_width, label=f'Soil {result["field"]}')

    # max_x_value = len(irrigation_data)
    plt.xticks(np.arange(0, 19, 1))
    plt.xlabel('Week')
    plt.ylabel('Irrigation (mm)')
    plt.title('Weekly Irrigation Amounts for All Fields (Bar Graph)')
    plt.grid(True)
    plt.legend(title='Soil Values')

    # Show the plot
    plt.show()
# plot_daily_irrigation(results=results)
plot_weekly_irrigation(results=results)

def combine_results(results, num_ha=1):
    total_yield = 0
    total_irr = 0
    for result in results:
        total_yield += result['final_stats']['Dry yield (tonne/ha)'][0]
        total_irr += result['final_stats']['Seasonal irrigation (mm)'][0]
    return total_yield, total_irr

print(results[0]['final_stats'])

total_yield, total_irr = combine_results(results=results)
print(f'The final yield is {total_yield} and the final seasonal irrigation is {total_irr}')

exit()


results_df = pd.DataFrame(columns=['Run', 'Dry yield (tonne/ha)', 'Seasonal Irrigation (mm)'])
# results_df = pd.DataFrame(columns=['Run', 'Yield Potential (tonne/ha)', 'Seasonal Irrigation (mm)'])

i = 15
while i <= 25:
    i = round(i,2)
    water_flux, final_stats = run_simulation(moisture_threshold=i)
    print(final_stats)
    # print(final_stats)
    new_row = pd.DataFrame({'Run':f'{i}',
                            'Dry yield (tonne/ha)': [final_stats['Dry yield (tonne/ha)'][0]],
                            'Seasonal Irrigation (mm)': [final_stats['Seasonal irrigation (mm)'][0]]})
    # new_row = pd.DataFrame({'Run':f'{i}',
    #                         'Yield potential (tonne/ha)': [final_stats['Yield potential (tonne/ha)'][0]],
    #                         'Seasonal Irrigation (mm)': [final_stats['Seasonal irrigation (mm)'][0]]})
    results_df = pd.concat([results_df, new_row], ignore_index=True)
    i += 0.1

# Plot the results
fig, ax = plt.subplots(2, 1, figsize=(10, 10))

# Plot yield for each run
ax[0].plot(results_df['Run'].to_numpy(), results_df['Dry yield (tonne/ha)'].to_numpy(), marker='o', linestyle='-', color='b')
ax[0].set_ylim(10,15)
# ax[0].plot(results_df['Run'].to_numpy(), results_df['Yield potential (tonne/ha)'].to_numpy(), marker='o', linestyle='-', color='b')
ax[0].set_title('Yield for each run')
# ax[0].set_xlabel('Moisture Threshold (Volumetric Water Content)')
ax[0].set_xlabel('Moisture Threshold (%)')
ax[0].set_xticklabels(results_df['Run'].to_numpy()[::5])
ax[0].set_xticks(results_df['Run'].to_numpy()[::5])
ax[0].set_ylabel('Dry yield (tonne/ha)')

# Plot seasonal irrigation for each run
ax[1].plot(results_df['Run'].to_numpy(), results_df['Seasonal Irrigation (mm)'].to_numpy(), marker='o', linestyle='-', color='g')
ax[1].set_title('Seasonal Irrigation for each run')
# ax[1].set_xlabel('Moisture Threshold (Volumetric Water Content)')
ax[1].set_xlabel('Moisture Threshold (%)')
ax[1].set_xticklabels(results_df['Run'].to_numpy()[::5])
ax[1].set_xticks(results_df['Run'].to_numpy()[::5])
ax[1].set_ylabel('Seasonal Irrigation (mm)')

plt.tight_layout()
plt.show()



#Plot the irrigation of each individual day

# irrigation_data = water_flux[['time_step_counter', 'IrrDay']]
# mask = (irrigation_data['time_step_counter'] == 0) & (irrigation_data['time_step_counter'].index != 0)

# Find the index of the first occurrence of this condition
# first_zero_index = irrigation_data[mask].index.min()

# # Drop all rows from this index onwards if such an index exists
# if pd.notna(first_zero_index):
#     irrigation_data = irrigation_data.iloc[:first_zero_index]
# print(irrigation_data)

# print(irrigation_data.columns)
# # Plot the daily irrigation amounts
# plt.figure(figsize=(12, 6))
# plt.plot(irrigation_data['time_step_counter'].to_numpy(), irrigation_data['IrrDay'].to_numpy(), marker='o')
# plt.xlabel('Day')
# plt.ylabel('Irrigation (mm)')
# plt.title('Daily Irrigation Amounts')
# plt.grid(True)
# plt.show()