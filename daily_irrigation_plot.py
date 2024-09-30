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
import seaborn as sns
import numpy as np

path = get_filepath('champion_climate.txt')
wdf = prepare_weather(path)

sim_start = '1982/05/01'
# sim_end = '2018/10/30'
sim_end = '1983/05/01'

soil= Soil('SandyLoam')
crop = Crop('Maize',planting_date='05/01')
initWC = InitialWaterContent(value=['FC'])

# create an irrigation schedule consisting of the first Tuesday of each month
all_days = pd.date_range(sim_start,sim_end) # list of all dates in simulation period
new_month=True
dates=[]

# iterate through all simulation days
for date in all_days:
    dates.append(date.date())
depths = [25]*len(dates) # depth of irrigation applied
schedule=pd.DataFrame([dates,depths]).T # create pandas DataFrame
schedule.columns=['Date','Depth'] # name columns
# print(schedule)

scheduled = IrrigationManagement(irrigation_method=3, Schedule=schedule)
rainfed = IrrigationManagement(irrigation_method=0)
crop.Name = 'Scheduled'
model = AquaCropModel(sim_start,
                    sim_end,
                    wdf,
                    soil,
                    crop,
                    initial_water_content=initWC,
                    irrigation_management=scheduled) # create model

model.run_model(till_termination=True)

water_flux = model._outputs.water_flux
if isinstance(water_flux, np.ndarray):
    columns = ['time_step_counter', 'season_counter', 'dap', 'Wr', 'z_gw', 'surface_storage', 
               'IrrDay', 'Infl', 'Runoff', 'DeepPerc', 'CR', 'GwIn', 'Es', 'EsPot', 'Tr', 'TrPot']
    water_flux = pd.DataFrame(model._outputs.water_flux, columns=columns)

print(water_flux)
# Extract the daily irrigation data from the model outputs
irrigation_data = water_flux[['time_step_counter', 'IrrDay']]
mask = (irrigation_data['time_step_counter'] == 0) & (irrigation_data['time_step_counter'].index != 0)

# Find the index of the first occurrence of this condition
first_zero_index = irrigation_data[mask].index.min()

# Drop all rows from this index onwards if such an index exists
if pd.notna(first_zero_index):
    irrigation_data = irrigation_data.iloc[:first_zero_index]
print(irrigation_data)

print(irrigation_data.columns)
# Plot the daily irrigation amounts
plt.figure(figsize=(12, 6))
plt.plot(irrigation_data['time_step_counter'].to_numpy(), irrigation_data['IrrDay'].to_numpy(), marker='o')
plt.xlabel('Day')
plt.ylabel('Irrigation (mm)')
plt.title('Daily Irrigation Amounts')
plt.grid(True)
plt.show()
