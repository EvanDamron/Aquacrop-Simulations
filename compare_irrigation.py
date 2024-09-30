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

def create_irrigation_schedule(sim_start, sim_end, irrigation_prob):
    # List of all days in the simulation period
    all_days = pd.date_range(sim_start, sim_end)
    
    # Determine the number of days to irrigate based on the probability
    num_days_with_irrigation = int(irrigation_prob * len(all_days))
    
    # Spread the irrigation days evenly across the simulation period
    irrigated_days = np.linspace(0, len(all_days) - 1, num_days_with_irrigation, dtype=int)
    
    depths = [0] * len(all_days)
    for day in irrigated_days:
        depths[day] = 25  # Apply 25mm of irrigation on the chosen days
    
    schedule = pd.DataFrame({'Date': all_days.date, 'Depth': depths})
    return schedule

def run_simulation(irrigation_prob):
    # Prepare weather data
    path = get_filepath('champion_climate.txt')
    wdf = prepare_weather(path)

    sim_start = '1982/05/01'
    sim_end = '1983/05/01'
    soil = Soil('SandyLoam')
    crop = Crop('Maize', planting_date='05/01')
    initWC = InitialWaterContent(value=['FC'])

    # Create irrigation schedule for the entire season based on the probability
    schedule = create_irrigation_schedule(sim_start, sim_end, irrigation_prob)
    print(schedule)

    # Create and run the model
    scheduled = IrrigationManagement(irrigation_method=3, Schedule=schedule)
    model = AquaCropModel(sim_start,
                        sim_end,
                        wdf,
                        soil,
                        crop,
                        initial_water_content=initWC,
                        irrigation_management=scheduled)
    
    model.run_model(till_termination=True)

    # Get the final output
    final_water_flux = model._outputs.water_flux
    final_stats = model._outputs.final_stats
    final_water_storage = model._outputs.water_storage
    return final_water_flux, final_stats, final_water_storage

results_df = pd.DataFrame(columns=['Run', 'Dry yield (tonne/ha)', 'Seasonal Irrigation (mm)'])

i = 0
while i <= 1:
    i = round(i, 2)
    water_flux, final_stats, water_storage = run_simulation(irrigation_prob=i)
    
    new_row = pd.DataFrame({'Run':f'{i}',
                            'Dry yield (tonne/ha)': [final_stats['Dry yield (tonne/ha)'][0]],
                            'Seasonal Irrigation (mm)': [final_stats['Seasonal irrigation (mm)'][0]]})
    
    results_df = pd.concat([results_df, new_row], ignore_index=True)
    i += 0.01

# Plot the results
fig, ax = plt.subplots(2, 1, figsize=(10, 10))

# Plot yield for each run
ax[0].plot(results_df['Run'].to_numpy(), results_df['Dry yield (tonne/ha)'].to_numpy(), marker='o', linestyle='-', color='b')
ax[0].set_ylim(10,15)
ax[0].set_title('Yield for each run')
ax[0].set_xlabel('Simulation Run')
ax[0].set_xticklabels(results_df['Run'].to_numpy()[::5])
ax[0].set_xticks(results_df['Run'].to_numpy()[::5])
ax[0].set_ylabel('Dry yield (tonne/ha)')

print("irrigation results:")
print(results_df['Seasonal Irrigation (mm)'])

# Plot seasonal irrigation for each run
ax[1].plot(results_df['Run'].to_numpy(), results_df['Seasonal Irrigation (mm)'].to_numpy(), marker='o', linestyle='-', color='g')
ax[1].set_title('Seasonal Irrigation for each run')
ax[1].set_xlabel('Simulation Run')
ax[1].set_xticklabels(results_df['Run'].to_numpy()[::5])
ax[1].set_xticks(results_df['Run'].to_numpy()[::5])
ax[1].set_ylabel('Seasonal Irrigation (mm)')

plt.tight_layout()
plt.show()
