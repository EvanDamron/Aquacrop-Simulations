from simulation_class import Simulation, run_simulations_get_moisture, run_final_simulation_threads
import threading
import queue
import pandas as pd

def main():
    soils = []
    