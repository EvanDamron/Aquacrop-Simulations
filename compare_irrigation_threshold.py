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


def run_simulation(irrigation_threshold):
    # Prepare weather data
    path = get_filepath('champion_climate.txt')
    wdf = prepare_weather(path)

    sim_start = '1982/05/01'
    sim_end = '1983/05/01'
    soil = Soil('SandyLoam')
    crop = Crop('Maize', planting_date='05/01')
    initWC = InitialWaterContent(value=['FC'])

    # Create and run the model
    irrigation = IrrigationManagement(irrigation_method=4, NetIrrSMT=irrigation_threshold)
    model = AquaCropModel(sim_start,
                        sim_end,
                        wdf,
                        soil,
                        crop,
                        initial_water_content=initWC,
                        irrigation_management=irrigation)
    
    model.run_model(till_termination=True)

    # Get the final output
    final_water_flux = model._outputs.water_flux
    final_stats = model._outputs.final_stats
    return final_water_flux, final_stats

results_df = pd.DataFrame(columns=['Run', 'Dry yield (tonne/ha)', 'Seasonal Irrigation (mm)'])

i = 0
while i <= 50:
    i = round(i, 1)
    water_flux, final_stats = run_simulation(irrigation_threshold=i)
    
    new_row = pd.DataFrame({'Run':f'{i}',
                            'Dry yield (tonne/ha)': [final_stats['Dry yield (tonne/ha)'][0]],
                            'Seasonal Irrigation (mm)': [final_stats['Seasonal irrigation (mm)'][0]]})
    
    results_df = pd.concat([results_df, new_row], ignore_index=True)
    i += 0.5

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

# Plot seasonal irrigation for each run
ax[1].plot(results_df['Run'].to_numpy(), results_df['Seasonal Irrigation (mm)'].to_numpy(), marker='o', linestyle='-', color='g')
ax[1].set_title('Seasonal Irrigation for each run')
ax[1].set_xlabel('Simulation Run')
ax[1].set_xticklabels(results_df['Run'].to_numpy()[::5])
ax[1].set_xticks(results_df['Run'].to_numpy()[::5])
ax[1].set_ylabel('Seasonal Irrigation (mm)')

plt.tight_layout()
plt.show()
