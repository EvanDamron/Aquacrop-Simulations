import os
os.environ['DEVELOPMENT'] = 'DEVELOPMENT'  # Run purely in python

from simulation_class import Simulation
import threading
import queue
import pandas as pd

def get_training_data():
    simulations = []
    sand = 10
    sim_id = 0
    start_date = '1982/05/01'

    for clay in range(10, 91, 10):
        silt = 100 - sand - clay
        print(f'Creating simulation for soil type ({sand}, {silt}, {clay})')
        sim = Simulation((sand, clay, silt), start_date, sim_id)
        simulations.append(sim)
        sim_id += 1

    schedules = [pd.DataFrame(columns=['Date', 'Depth']) for _ in range(len(simulations))]

if __name__ == "__main__":
    main()
