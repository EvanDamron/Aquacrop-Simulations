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
import threading
import queue

class Simulation:
    def __init__(self, soil_values, sim_start, id) -> None:
        self.id = id
        self.soil = Soil('custom')
        self.soil.add_layer_from_texture(thickness=self.soil.zSoil,
                                Sand=soil_values[0],Clay=soil_values[1],
                                OrgMat=2.5,penetrability=100)
        self.SAT = self.soil.profile['th_s'][0]
        path = get_filepath('champion_climate.txt')
        self.wdf = prepare_weather(path)
        self.initWC = InitialWaterContent(value=['FC'])
        self.sim_start = sim_start
        self.crop = Crop('Maize', planting_date=pd.to_datetime(sim_start).strftime('%m/%d'))

    def run_sim(self, schedule, end_date):
        irrigation = IrrigationManagement(irrigation_method=3, Schedule=schedule)
        model = AquaCropModel(self.sim_start,
                              end_date.strftime('%Y/%m/%d'),
                              self.wdf,
                              self.soil,
                              self.crop,
                              self.initWC,
                              irrigation) 
        model.run_model(till_termination=True)
        return model._outputs.water_storage, model._outputs.final_stats
    
if __name__ == "__main__":
    start_date = '1982/05/01'
    end_date = '1983/05/01'

    simulations = []
    sand = 10
    sim_id = 0
    for clay in range(10, 91, 10):
        silt = 100 - sand - clay
        simulations.append(Simulation((sand, clay, silt), start_date, sim_id))
        sim_id += 1
    # simulations[0].run_sim()
    
    current_end = pd.to_datetime(start_date) + pd.Timedelta(days=7)

    def run_simulation_in_thread(sim, q, schedule, end):
        water_storage, _ = sim.run_sim(schedule, end)
        # Traverse the dataframe backwards to get the last day's moisture value
        for i in range(len(water_storage) - 1, 0, -1):
            if water_storage['growing_season'].iloc[i] == 1:
                moisture_percentage = water_storage['th1'].iloc[i] / sim.SAT
                break
        q.put((sim.id, moisture_percentage))


    result_queue = queue.Queue()
    threads = []
    schedules = [pd.DataFrame(columns=['Date', 'Depth'])] * len(simulations)  # Initial empty schedule
    for i, simulation in enumerate(simulations):
        thread = threading.Thread(target=run_simulation_in_thread, args=(simulation, result_queue, schedules[i], current_end))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    moisture_levels = [None] * len(simulations)
    while not result_queue.empty():
        sim_id, moisture = result_queue.get()
        moisture_levels[sim_id] = moisture
    
    print(moisture_levels)
    irrigate = [1 if x < 0.71 else 0 for x in moisture_levels]
    print(irrigate)
    for i, val in enumerate(irrigate):
        if val == 1:
            print(f'field {i} is below the irrigation threshold. Irrigating.')
            new_week = pd.DataFrame({
                'Date' : [current_end + pd.Timedelta(days=7)],
                'Depth' : [25]
            })
            schedules[i] = pd.concat([schedules[i], new_week], ignore_index=True)
        print(schedules[i])

