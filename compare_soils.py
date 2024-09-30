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

def run_simulation(sand, clay):
    # Prepare weather data
    path = get_filepath('champion_climate.txt')
    wdf = prepare_weather(path)

    sim_start = '1982/05/01'
    sim_end = '1983/05/01'
    # soil = Soil('SandyLoam', cn=curve_number, rew=7)
    soil = Soil('custom')
    soil.add_layer_from_texture(thickness=soil.zSoil,
                              Sand=sand,Clay=clay,
                              OrgMat=2.5,penetrability=100)
    # print(soil.profile.head(1))
    # soil = Soil(soil_type=soil_type)
    crop = Crop('Maize', planting_date='05/01')
    initWC = InitialWaterContent(value=['FC'])

    # Create and run the model
    rainfed = IrrigationManagement(irrigation_method=0)
    # threshold = IrrigationManagement(irrigation_method=4, NetIrrSMT=50)
    model = AquaCropModel(sim_start,
                        sim_end,
                        wdf,
                        soil,
                        crop,
                        initial_water_content=initWC,
                        irrigation_management=rainfed)
    
    model.run_model(till_termination=True)

    # Get the final output
    final_water_flux = model._outputs.water_flux
    final_stats = model._outputs.final_stats
    final_water_storage = model._outputs.water_storage
    return final_water_flux, final_stats, final_water_storage

results_df = pd.DataFrame(columns=['Run', 'Dry yield (tonne/ha)', 'Seasonal Irrigation (mm)'])

combinations = []
# for first in range(10, 91, 10):
first = 10
for second in range(10, 91-first, 10):
    combinations.append((first, second, 100 - first - second))

fig, ax = plt.subplots(2, 1, figsize=(10, 10))
# moisture_dfs = []
for combo in combinations:
    sand = combo[0]
    clay = combo[1]
    silt = combo[2]
    # i = round(i, 2)
    water_flux, final_stats, water_storage = run_simulation(sand, clay)

    # soil = Soil('custom')
    # soil.add_layer_from_texture(thickness=soil.zSoil,
    #                           Sand=sand,Clay=clay,
    #                           OrgMat=2.5,penetrability=100)
    # sat = soil.profile['th_s'][0]
    water_storage = water_storage[water_storage['growing_season'] == 1]
    dap = water_storage['dap']
    th1 = water_storage['th1']
    ax[1].plot(dap.to_numpy(), th1.to_numpy(), label=str(combo))
    # print(water_flux.head(30))
    # print(water_storage.head(30))
    # exit()
    # moisture_dfs.append(water_storage[['dap', 'th1']])
    new_row = pd.DataFrame({'Run':f'({sand}, {clay}, {silt})',
                            'Dry yield (tonne/ha)': [final_stats['Dry yield (tonne/ha)'][0]],
                            'Seasonal Irrigation (mm)': [final_stats['Seasonal irrigation (mm)'][0]]})
    
    results_df = pd.concat([results_df, new_row], ignore_index=True)

# soil_types = ['SandyLoam']
# for soil in soil_types:
#     water_flux, final_stats, water_storage = run_simulation(soil_type=soil)
    
#     new_row = pd.DataFrame({'Run':f'{i}',
#                             'Dry yield (tonne/ha)': [final_stats['Dry yield (tonne/ha)'][0]],
#                             'Seasonal Irrigation (mm)': [final_stats['Seasonal irrigation (mm)'][0]]})
    
#     results_df = pd.concat([results_df, new_row], ignore_index=True)
# Plot the results

# Plot yield for each run
ax[0].plot(results_df['Run'].to_numpy(), results_df['Dry yield (tonne/ha)'].to_numpy(), marker='o', linestyle='-', color='b')
# ax[0].set_ylim(10,15)
ax[0].set_title('Yield for each run')
ax[0].set_xlabel('Sand, Clay, Silt')
ax[0].tick_params(axis='x', labelsize=5)
# ax[0].set_xticklabels(results_df['Run'].to_numpy()[::5])
# ax[0].set_xticks(results_df['Run'].to_numpy()[::5])
ax[0].set_ylabel('Dry yield (tonne/ha)')


# print("irrigation results:")
# print(results_df['Seasonal Irrigation (mm)'])

# for i, combo in enumerate(combinations):
#     df = moisture_dfs[i]
#     dap = df['dap']
#     th1 = df['th1']
#     ax[1].plot(dap.to_numpy(), th1.to_numpy(), label=str(combo))
ax[1].set_title('Daily Moisture Level (%) for each run')
ax[1].set_xlabel('Days After Planting')
# ax[1].set_xticklabels(fontsize=6)
# ax[1].tick_params(axis='x', labelsize=5)
# ax[1].set_xticklabels(results_df['Run'].to_numpy()[::5])
# ax[1].set_xticks(results_df['Run'].to_numpy()[::5])
ax[1].set_ylabel('Moisture Level')
ax[1].legend()


# Plot seasonal irrigation for each run
# ax[1].plot(results_df['Run'].to_numpy(), results_df['Seasonal Irrigation (mm)'].to_numpy(), marker='o', linestyle='-', color='g')
# ax[1].set_title('Seasonal Irrigation for each run')
# ax[1].set_xlabel('Sand, Clay Silt')
# # ax[1].set_xticklabels(fontsize=6)
# ax[1].tick_params(axis='x', labelsize=5)
# # ax[1].set_xticklabels(results_df['Run'].to_numpy()[::5])
# # ax[1].set_xticks(results_df['Run'].to_numpy()[::5])
# ax[1].set_ylabel('Seasonal Irrigation (mm)')

plt.tight_layout()
plt.show()
