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

def run_simulation(moisture_threshold):

    # Prepare weather data
    path = get_filepath('champion_climate.txt')
    wdf = prepare_weather(path)

    sim_start = '1982/05/01'
    sim_end = '1983/05/01'
    soil = Soil('SandyLoam')
    # For when moisture_threshold is a percentage, not an actual moisture value
    SAT = soil.profile['th_s'][0]
    crop = Crop('Maize', planting_date='05/01')
    initWC = InitialWaterContent(value=['FC'])

    # Initialize an empty schedule
    schedule = pd.DataFrame(columns=['Date', 'Depth'])

    # Set the simulation end date for weekly simulation
    current_end = pd.to_datetime(sim_start) + pd.Timedelta(days=7)

    # Run the simulation week by week
    while current_end <= pd.to_datetime(sim_end):
        # print(f"Running simulation up to: {current_end.date()}")
        
        # Create model with the current schedule
        scheduled = IrrigationManagement(irrigation_method=3, Schedule=schedule)
        model = AquaCropModel(sim_start,
                            current_end.strftime('%Y/%m/%d'),
                            wdf,
                            soil,
                            crop,
                            initial_water_content=initWC,
                            irrigation_management=scheduled)
        
        # Run the model
        model.run_model(till_termination=True)
        
        # Get the output for the past week
        water_storage = model._outputs.water_storage

        counter = 0
        # Traverse the dataframe backwards until we find the last day of the growing season and store it's moisture value
        for i in range(len(water_storage) - 1, -1, -1):
            counter += 1
            if water_storage['growing_season'].iloc[i] == 1:
                current_moisture = water_storage['th1'].iloc[i]
                break
        
        current_moisture_percentage = (current_moisture / SAT) * 100
        # Decide on the irrigation management for the next week
        next_week_dates = pd.date_range(current_end + pd.Timedelta(days=1), 
                                        current_end + pd.Timedelta(days=7))
        next_week_depths = [0] * len(next_week_dates)
        
        # # When the moisture threshold is specified as a mm3/mm3 value
        # if current_moisture < moisture_threshold:
        #     next_week_depths[0] = 25

        # When the moisture threshold is specified as a percentage
        if current_moisture_percentage < moisture_threshold:
            next_week_depths[0] = 25
        new_entries = pd.DataFrame({'Date': next_week_dates.date, 'Depth': next_week_depths})
        
        # Update the irrigation schedule
        schedule = pd.concat([schedule, new_entries])
        
        if counter > 2:
            break
        # Move to the next week
        current_end += pd.Timedelta(days=7)

    # Final run for the entire season
    final_scheduled = IrrigationManagement(irrigation_method=3, Schedule=schedule)
    model = AquaCropModel(sim_start,
                        sim_end,
                        wdf,
                        soil,
                        crop,
                        initial_water_content=initWC,
                        irrigation_management=final_scheduled)
    model.run_model(till_termination=True)

    # Get the final output
    final_water_flux = model._outputs.water_flux
    final_stats = model._outputs.final_stats
    return final_water_flux, final_stats

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