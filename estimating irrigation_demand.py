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

path = get_filepath('champion_climate.txt')
wdf = prepare_weather(path)

sim_start = '1982/05/01'
sim_end = '2018/10/30'

soil= Soil('SandyLoam')
crop = Crop('Maize',planting_date='05/01')
initWC = InitialWaterContent(value=['FC'])

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
            # dates.append(date)
            dates.append(date.date())
            new_month=False
depths = [25]*len(dates) # depth of irrigation applied
schedule=pd.DataFrame([dates,depths]).T # create pandas DataFrame
schedule.columns=['Date','Depth'] # name columns
print(schedule)

scheduled = IrrigationManagement(irrigation_method=3, Schedule=schedule)
# scheduled = IrrigationManagement(irrigation_method=3)
rainfed = IrrigationManagement(irrigation_method=0)

labels = ['rainfed', 'scheduled']
strategies = [rainfed, scheduled]
outputs = []

for i, irr_mngt in enumerate(strategies):
    crop.Name = labels[i]
    model = AquaCropModel(sim_start,
                        sim_end,
                        wdf,
                        soil,
                        crop,
                        initial_water_content=initWC,
                        irrigation_management=irr_mngt) # create model
    model.run_model(till_termination=True) # run model till the end
    # print(model._outputs.final_stats.head(10))
    outputs.append(model._outputs.final_stats) # save results

print(outputs[0].columns)
print(outputs[0])
print(outputs[1])

# Create a figure with two subplots for the line plots
fig, ax = plt.subplots(2, 1, figsize=(12, 10))

# Plot yield for each season for rainfed and scheduled
for i in range(len(outputs)):
    ax[0].plot(outputs[i]['Season'].values, outputs[i]['Yield potential (tonne/ha)'].values, label=labels[i])
    ax[1].plot(outputs[i]['Season'].values, outputs[i]['Seasonal irrigation (mm)'].values, label=labels[i])

# Set labels and titles
ax[0].set_xlabel('Season', fontsize=14)
ax[0].set_ylabel('Yield potential (tonne/ha)', fontsize=14)
ax[0].set_title('Yield Potential Over Seasons', fontsize=16)
ax[0].legend(title="Irrigation Strategy")

ax[1].set_xlabel('Season', fontsize=14)
ax[1].set_ylabel('Seasonal Irrigation (mm)', fontsize=14)
ax[1].set_title('Seasonal Irrigation Over Seasons', fontsize=16)
ax[1].legend(title="Irrigation Strategy")

plt.tight_layout()
plt.show()