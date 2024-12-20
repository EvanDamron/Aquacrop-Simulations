import os
os.environ['DEVELOPMENT'] = 'DEVELOPMENT'  # Run purely in python

# suppress warning messages
import logging
import warnings
logging.getLogger().setLevel(logging.ERROR)
warnings.simplefilter(action='ignore', category=FutureWarning)

from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, IrrigationManagement
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
        # self.initWC = InitialWaterContent(value=['FC'])
        self.initWC = InitialWaterContent(wc_type='Num', value=[0.2])
        self.sim_start = sim_start.strftime('%Y/%m/%d')
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
        return model._outputs.water_storage, model._outputs.final_stats, model._outputs.water_flux
    
# Test functionality
if __name__ == "__main__":
    start_date = '1982/05/01'
    end_date = '1983/05/01'
    current_end = pd.to_datetime(start_date) + pd.Timedelta(days=7)

    simulations = []
    sand = 20
    sim_id = 0
    for clay in range(20, 61, 5):
        silt = 100 - sand - clay
        print(f'Creating simulation for soil type ({sand}, {silt}, {clay})')
        simulations.append(Simulation((sand, clay, silt), start_date, sim_id))
        sim_id += 1

    schedules = [pd.DataFrame(columns=['Date', 'Depth']) for _ in range(len(simulations))]

    def run_simulations_get_moisture(sim, q, schedule, end):
        try:
            water_storage, _ = sim.run_sim(schedule, end)
            q.put((sim.id, water_storage))
        except Exception as e:
            print(f'Error running simulation {sim.id}: {e}')
            q.put((sim.id, None))
            exit(1)

    harvested = False
    while current_end < pd.to_datetime(end_date):
        if harvested:
            break
        print(f'Running simulation for week ending {current_end}')
        result_queue = queue.Queue()
        threads = []
        for i, simulation in enumerate(simulations):
            thread = threading.Thread(target=run_simulations_get_moisture, args=(simulation, result_queue, schedules[i], current_end))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        moisture_levels = [None] * len(simulations)
        while not result_queue.empty():
            sim_id, water_storage = result_queue.get()
            # Traverse the dataframe backwards to get the last day's moisture value
            for i in range(len(water_storage) - 1, 0, -1):
                if water_storage['growing_season'].iloc[i] == 1:
                    if i < len(water_storage) - 2:   # Check for more than one empty day at end of dataframe
                        harvested = True
                    moisture_percentage = water_storage['th1'].iloc[i] / simulations[sim_id].SAT
                    break
            moisture_levels[sim_id] = moisture_percentage
        
        irrigate = [1 if x < 0.7 else 0 for x in moisture_levels]
        for i, val in enumerate(irrigate):
            if val == 1:
                new_week = pd.DataFrame({
                    'Date' : [current_end + pd.Timedelta(days=7)],
                    'Depth' : [25]
                })
                schedules[i] = pd.concat([schedules[i], new_week], ignore_index=True)
        
        irrigated_fields = [i for i, val in enumerate(irrigate) if val == 1]
        print(f'Irrigated fields for week ending {current_end}: {", ".join(map(str, irrigated_fields))}')
        current_end += pd.Timedelta(days=7)
        print('----------------------------------------')
    print('Running final simulations')

    def run_final_simulation_threads(sim, q, schedule, end):
        try:
            final_water_storage, final_results = sim.run_sim(schedule, end)
            q.put((sim.id, final_results, final_water_storage))
        except Exception as e:
            print(f'Error running simulation {sim.id}: {e}')
            q.put((sim.id, None, None))

    result_queue = queue.Queue()
    threads = []
    for i, simulation in enumerate(simulations):
        thread = threading.Thread(target=run_final_simulation_threads, args=(simulation, result_queue, schedules[i], pd.to_datetime(end_date)))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()

    final_results = [None] * len(simulations)
    final_storage = [None] * len(simulations)
    while not result_queue.empty():
        sim_id, results, water_storage = result_queue.get()
        final_results[sim_id] = results
        final_storage[sim_id] = water_storage

    pd.set_option('display.max_rows', 20)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    for i, results in enumerate(final_results):
        print(f'Simulation {i} results:')
        print(results)
        # print('----------------------------------------')
        # print(f'Simulation {i} water storage:')
        # print(final_storage[i])

    print('All simulations done.')
    
