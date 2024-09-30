import os
os.environ['DEVELOPMENT'] = 'DEVELOPMENT'

from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, FieldMngt, GroundWater, IrrigationManagement
from aquacrop.utils import prepare_weather, get_filepath

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# get location of built in weather data file
path = get_filepath('tunis_climate.txt')
# read in weather data file and put into correct format
wdf = prepare_weather(path)

print('\n')
# show weather data
# print(wdf.head())

sim_start = "1979/10/01"
sim_end = "1985/05/30"

# create an irrigation schedule consisting of the first Tuesday of each month
all_days = pd.date_range(sim_start,sim_end) # list of all dates in simulation period
new_month=True
dates=[]
# iterate through all simulation days
for date in all_days:
    #check if new month
    if date.is_month_start:
        new_month=True

    if new_month:
        # check if tuesday (dayofweek=1)
        if date.dayofweek==1:
            #save date
            dates.append(date)
            new_month=False


depths = [25]*len(dates) # depth of irrigation applied
schedule=pd.DataFrame([dates,depths]).T # create pandas DataFrame
schedule.columns=['Date','Depth'] # name columns
# print(schedule)


model_os = AquaCropModel(
            sim_start_time= sim_start,
            sim_end_time= sim_end,
            weather_df=wdf,
            soil=Soil(soil_type='SandyLoam'),
            crop=Crop('Wheat', planting_date='10/01'),
            initial_water_content=InitialWaterContent(value=['FC']),
            irrigation_management=IrrigationManagement(irrigation_method=3, schedule=schedule)
        )
model_os.run_model(till_termination=True)

# # Save water flux data to a CSV file
# model_os._outputs.water_flux.head(50).to_csv('water_flux_output.csv', index=False)

# # Save water storage data to a CSV file
model_os._outputs.water_storage.to_csv('water_storage_output.csv', index=False)


# print(model_os._outputs.final_stats.head())
# print(model_os._outputs.water_flux.head(50))
print(model_os._outputs.final_stats.head(50))
# print(model_os._outputs.crop_growth)

model_os = AquaCropModel(
            sim_start_time= sim_start,
            sim_end_time= sim_end,
            weather_df=wdf,
            soil=Soil(soil_type='SandyLoam'),
            crop=Crop('Wheat', planting_date='10/01'),
            initial_water_content=InitialWaterContent(value=['FC']),
            irrigation_management=IrrigationManagement(irrigation_method=0)
        )
model_os.run_model(till_termination=True)
print(model_os._outputs.final_stats.head(50))