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
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def run_simulations(soil_values, start_date='1982/05/01', end_date='1983/05/01', threshold=0.5):
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
        
        irrigate = [1 if x < threshold else 0 for x in moisture_levels]
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


def run_simulations_predictive(soil_values, visited_fields, model, start_date='1983/05/01', end_date='1984/05/01', threshold=0.5):
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
    mse_vals = []
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
        
        moisture_levels_actual = [None] * len(simulations)
        while not result_queue.empty():
            sim_id, water_storage = result_queue.get()
            # Traverse the dataframe backwards to get the last day's moisture value
            for i in range(len(water_storage) - 1, 0, -1):
                if water_storage['growing_season'].iloc[i] == 1:
                    if i < len(water_storage) - 2:   # Check for more than one empty day at end of dataframe
                        harvested = True
                    moisture_percentage = water_storage['th1'].iloc[i] / simulations[sim_id].SAT
                    break
            moisture_levels_actual[sim_id] = moisture_percentage
        
        input_features = [[moisture_levels_actual[i] for i in visited_fields]]
        input_features_df = pd.DataFrame(input_features, columns=[f'sensor_{i}' for i in visited_fields])
        moisture_levels_predictions = model.predict(input_features_df)
        moisture_levels_predicted = [None] * len(soil_values)
        for position, index in enumerate(visited_fields):
            moisture_levels_predicted[index] = moisture_levels_actual[position]
        predicted_index = 0
        for i in range(len(moisture_levels_predicted)):
            if moisture_levels_predicted[i] is None:
                # Replace None with the predicted value, keeping track of the order in moisture_levels_predictions
                moisture_levels_predicted[i] = moisture_levels_predictions[0][predicted_index]
                predicted_index += 1 

        actual = [moisture_levels_actual[i] for i in range(len(moisture_levels_actual)) if i not in visited_fields]
        mse = mean_squared_error(actual, moisture_levels_predictions[0])
        mse_vals.append(mse)

        irrigate = [1 if x < threshold else 0 for x in moisture_levels_predicted]
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
    return final_results, final_storage, mse_vals


def get_training_data(water_storages):
    new_df = pd.DataFrame()
    for i, df in enumerate(water_storages):
        new_df[f'sensor_{i}'] = df['th1']
    new_df = new_df[(new_df != 0).any(axis=1)]
    return new_df


def get_soil_values(dim):
    # Define grid dimensions
    num_cols = dim  # Number of columns in the grid (fields horizontally)
    num_rows = dim  # Number of rows in the grid (fields vertically)

    # Initialize empty lists to store soil values
    sand_values = np.zeros((num_rows, num_cols), dtype=int)
    clay_values = np.zeros((num_rows, num_cols), dtype=int)
    silt_values = np.zeros((num_rows, num_cols), dtype=int)

    # Initialize the center or first field with random values
    center_sand = random.randint(20, 40)  # Example range for Sand
    center_clay = random.randint(20, 40)  # Example range for Clay
    center_silt = 100 - center_sand - center_clay

    # Initialize the center field (or start from any field)
    sand_values[0, 0] = center_sand
    clay_values[0, 0] = center_clay
    silt_values[0, 0] = center_silt

    # Define a function to generate neighboring values with smooth variations
    def generate_neighbor_values(sand, clay, variation=5):
        """Generate soil values similar to the neighboring field with small variation."""
        new_sand = np.clip(int(sand) + random.randint(-variation, variation), 0, 100)
        new_clay = np.clip(int(clay) + random.randint(-variation, variation), 0, 100)
        # Adjust silt to ensure the sum is still 100%
        new_silt = 100 - new_sand - new_clay
        return new_sand, new_clay, new_silt

    # Fill the grid with smooth variation from neighbors
    for i in range(num_rows):
        for j in range(num_cols):
            if i == 0 and j == 0:
                continue  # Skip the initialized center point
            if i == 0:
                # Horizontal neighbors only
                sand_values[i, j], clay_values[i, j], silt_values[i, j] = generate_neighbor_values(
                    sand_values[i, j-1], clay_values[i, j-1], silt_values[i, j-1]
                )
            elif j == 0:
                # Vertical neighbors only
                sand_values[i, j], clay_values[i, j], silt_values[i, j] = generate_neighbor_values(
                    sand_values[i-1, j], clay_values[i-1, j], silt_values[i-1, j]
                )
            else:
                # Average neighbors for smoothness
                avg_sand = (sand_values[i-1, j] + sand_values[i, j-1]) // 2
                avg_clay = (clay_values[i-1, j] + clay_values[i, j-1]) // 2
                avg_silt = 100 - avg_sand - avg_clay
                sand_values[i, j], clay_values[i, j], silt_values[i, j] = generate_neighbor_values(
                    avg_sand, avg_clay, avg_silt
                )

    # Now sand_values, clay_values, and silt_values have smoothly varying values for each grid cell

    # Example of how to store soil values in a list of dictionaries
    soil_values = []
    for i in range(num_rows):
        for j in range(num_cols):
            soil_values.append((sand_values[i, j], clay_values[i, j], silt_values[i, j]))

    return soil_values


def get_total_results(field_length, field_width, final_results):
    total_yield = 0
    total_irr = 0
    field_size_ha = field_length * field_width / 10000
    for i, results in enumerate(final_results):
        total_yield += results['Dry yield (tonne/ha)'][0] * field_size_ha
        total_irr += results['Seasonal irrigation (mm)'][0]
    average_irr = total_irr / len(final_results)
    return total_yield, average_irr


def train_model(x_train, y_train):
    """Train the model using the selected features and target sensors."""
    # Create pipeline for regression model
    if len(y_train.columns) == 1:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', TransformedTargetRegressor(
                regressor=GradientBoostingRegressor(random_state=42),
                transformer=StandardScaler()
            ))
        ])
    else:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', TransformedTargetRegressor(
                regressor=MultiOutputRegressor(GradientBoostingRegressor(random_state=42)),
                transformer=StandardScaler()
            ))
        ])
    # Train the model
    pipeline.fit(x_train, y_train)
    
    return pipeline





if __name__ == "__main__":
    # soil_values = get_soil_values(5)
    # for i in range(0, len(soil_values), 5):
    #     print(soil_values[i:i+5])
    random.seed(1)
    soil_values = []
    for i in range(25):
        soil_values.append((random.randint(10, 50), random.randint(20, 40), 100 - random.randint(20, 40)))
    
    # # for i in range(0, len(soil_values), 5):
    # #     print(soil_values[i:i+5])

    # final_results, final_storages = run_simulations(soil_values=soil_values, start_date='1983/05/01', end_date='1984/05/01', threshold=0.5)
    # total_yield, average_irr = get_total_results(field_length=200, field_width=200, final_results=final_results)
    # training_data = get_training_data(final_storages)
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.width', None)
    # pd.set_option('display.max_colwidth', None)
    # # training_data.to_csv('training_data.csv')
    # print(f'Total yield: {total_yield} tonnes')
    # print(f'Average irrigation: {average_irr} mm')


    # Now run the epsilon greedy algorithm to select sensors

    selected_sensors = [18, 15, 22, 6, 2, 1, 16, 17, 10, 9, 12, 8, 3, 21]
    training_data = pd.read_csv('training_data.csv')
    x_train = training_data[[f'sensor_{i}' for i in selected_sensors]]
    y_train = training_data[[f'sensor_{i}' for i in range(25) if i not in selected_sensors]]
    model = train_model(x_train, y_train)
    print('Model trained')
    # Now run the simulation with the trained model
    final_results, final_storages, mse_vals = run_simulations_predictive(soil_values=soil_values, visited_fields=selected_sensors, model=model)
    total_yield, average_irr = get_total_results(field_length=200, field_width=200, final_results=final_results)
    print(f'Total yield: {total_yield} tonnes')
    print(f'Average irrigation: {average_irr} mm')
    plt.plot(range(1, len(mse_vals) + 1), mse_vals, marker='o', linestyle='-', color='b')

    # Add titles and labels
    plt.title('MSE Values Over Time')
    plt.xlabel('Week')
    plt.ylabel('MSE')

    # Show the plot
    plt.grid(True)
    plt.show()
