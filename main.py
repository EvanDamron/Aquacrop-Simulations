import os
import random

from matplotlib import pyplot as plt
from shapely import Point, Polygon
os.environ['DEVELOPMENT'] = 'DEVELOPMENT'  # Run purely in python

from simulation_class import Simulation
import threading
import queue
import pandas as pd
import geopandas as gpd
import math
from epsilonGreedy import epsilonGreedy

def run_simulations(soil_values, start_date='1982/05/01', end_date='1983/05/01'):
    current_end = pd.to_datetime(start_date) + pd.Timedelta(days=7)
    simulations = []
    sim_id = 0
    for soil_value in soil_values:
        print(f'Creating simulation for soil type ({soil_value[0]}, {soil_value[1]}, {soil_value[2]})')
        simulations.append(Simulation((soil_value), start_date, sim_id))
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
    return final_results, final_storage


def get_training_data(water_storages):
    new_df = pd.DataFrame()
    for i, df in enumerate(water_storages):
        new_df[f'sensor_{i}'] = df['th1']
    new_df = new_df[(new_df != 0).any(axis=1)]
    return new_df

def create_map(drone_height, commRadius, plot_names, area_width=900, area_height=900):
    num_rows = int(math.sqrt(len(plot_names)))
    num_cols = int(math.sqrt(len(plot_names)))

    # Create grid cells (polygons)
    cell_width = area_width / num_cols
    cell_height = area_height / num_rows

    grid_cells = []
    sensor_locations = []

    for i in range(num_cols):
        for j in range(num_rows):
            # Define the corners of the grid cell
            x_min = i * cell_width
            x_max = (i + 1) * cell_width
            y_min = j * cell_height
            y_max = (j + 1) * cell_height
            # Create the cell as a polygon
            cell = Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])
            grid_cells.append(cell)
            
            # Randomly place a sensor location within the cell
            random_x = random.uniform(x_min, x_max)
            random_y = random.uniform(y_min, y_max)
            sensor_location = Point(random_x, random_y)
            sensor_locations.append(sensor_location)

    # Create a GeoDataFrame for the grid cells
    grid_gdf = gpd.GeoDataFrame(geometry=grid_cells)

    # Create a GeoDataFrame for the sensor locations
    sensor_gdf = gpd.GeoDataFrame(geometry=sensor_locations)
    sensor_gdf['Location'] = plot_names

    # Plot the grid and the sensor locations
    fig, ax = plt.subplots(figsize=(8, 8))
    grid_gdf.boundary.plot(ax=ax, color='black')  # Plot grid cell boundaries
    sensor_gdf.plot(ax=ax, color='blue', markersize=30)  # Plot sensor locations
    hoverPoints, sensorNames = getHoverPoints(sensor_gdf, commRadius, drone_height, ax)
    return fig, ax, hoverPoints, sensorNames, sensor_gdf

def getHoverPoints(sensors, commRadius, height, ax):
    # Add circles and find hover points
    droneRadius = (commRadius ** 2 - height ** 2) ** 0.5
    rangeCircles = sensors.copy()
    rangeCircles['Communication Range'] = droneRadius
    rangeCircles['geometry'] = sensors['geometry'].buffer(rangeCircles['Communication Range'])
    for circle in rangeCircles['geometry']:
        x, y = circle.exterior.xy
        vertices = list(zip(x, y))
        patch = plt.Polygon(vertices, edgecolor='black', facecolor='lime', alpha=0.4)
        ax.add_patch(patch)
    # find midpoints of overlapping sections and add them to hoverPoints gdf
    overlapsOf2 = gpd.overlay(df1=rangeCircles, df2=rangeCircles, how='intersection')
    overlapsOf3 = gpd.overlay(df1=overlapsOf2, df2=overlapsOf2, how='intersection')

    overlapsOf3['geometry_str'] = overlapsOf3['geometry'].astype(str)
    overlapsOf3 = overlapsOf3.drop_duplicates(subset='geometry_str').reset_index(drop=True)
    hoverPoints = gpd.GeoDataFrame(geometry=overlapsOf3['geometry'].centroid)
    hoverPoints['geometry_str'] = hoverPoints['geometry'].astype(str)
    hoverPoints = hoverPoints.drop_duplicates(subset='geometry_str').reset_index(drop=True)
    hoverPoints = hoverPoints.drop(columns=['geometry_str'])
    # create dictionary to correspond hover points to sensors
    sensorNames = {}
    for hoverPoint in hoverPoints['geometry']:
        sensorNames[hoverPoint] = []
        # if hoverPoint in sensors['geometry']:
        for circle in rangeCircles['geometry']:
            if hoverPoint.within(circle):
                sensorName = rangeCircles.loc[rangeCircles['geometry'] == circle, 'Location'].values[0]
                sensorNames[hoverPoint].append(sensorName)

    return hoverPoints, sensorNames


if __name__ == "__main__":
    soil_values = []
    sand = 20
    for clay in range(20, 61, 5):
        silt = 100 - sand - clay
        soil_values.append((sand, clay, silt))
    _, final_storages = run_simulations(soil_values=soil_values)
    training_data = get_training_data(final_storages)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(training_data)
    fig, ax, hoverPoints, sensorNames, sensor_gdf = create_map(10, 70, training_data.columns)
    epsilonGreedy(ax1=ax, HP_gdf=hoverPoints, )
    plt.show()

