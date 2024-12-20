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
from sklearn.linear_model import LinearRegression
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
            water_storage, _, _ = sim.run_sim(schedule, end)
            q.put((sim.id, water_storage))
        except Exception as e:
            print(f'Error running simulation {sim.id}: {e}')
            q.put((sim.id, None))
            exit(1)

    harvested = False
    while current_end < pd.to_datetime(end_date):
        if harvested:
            # if current_end.year != pd.to_datetime(end_date).year:
            #     current_end = pd.to_datetime(f'{current_end.year + 1}/05/01')
            #     harvested = False
            #     continue
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
            final_water_storage, final_results, final_flux = sim.run_sim(schedule, end)
            q.put((sim.id, final_results, final_water_storage, final_flux))
        except Exception as e:
            print(f'Error running simulation {sim.id}: {e}')
            q.put((sim.id, None, None, None))

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
    final_flux = [None] * len(simulations)
    while not result_queue.empty():
        sim_id, results, water_storage, water_flux = result_queue.get()
        final_results[sim_id] = results
        final_storage[sim_id] = water_storage
        final_flux[sim_id] = water_flux

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
    return final_results, final_storage, final_flux


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
            water_storage, _, _ = sim.run_sim(schedule, end)
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
                    moisture_percentage = water_storage['th1'].iloc[i]
                    break
            moisture_levels_actual[sim_id] = moisture_percentage
    

        input_features = [[moisture_levels_actual[i] for i in visited_fields]]  # Extract features for the visited fields
        input_features_df = pd.DataFrame(input_features, columns=[f'sensor_{i}' for i in visited_fields])
        # print(f'input features df: \n{input_features_df}')
        # print(model.named_steps['scaler'].feature_names_in_)
        # input_features_scaled = model.named_steps['scaler'].transform(input_features_df.to_numpy())
        # print(f'input features df: \n{input_features_df}')
        # print(f'input features scaled: \n{input_features_scaled}')
        # Make predictions using the model
        # predictions = model.predict(input_features_scaled)
        # print(f'input features df: \n{input_features_df}')
        predictions = model.predict(input_features_df)

        # print(f'Predicted moisture levels:\n {predictions}')

        # Create a list to store the predicted values
        new_moistures = [None] * len(moisture_levels_actual)

        # Insert actual values for visited fields
        for field_number in visited_fields:
            new_moistures[field_number] = moisture_levels_actual[field_number]

        # print(f'Values after inserting originals: \n {new_moistures}')

        # Insert predicted values for the unvisited fields
        temp = 0
        for i in range(len(new_moistures)):
            if new_moistures[i] is None:
                # Replace None with the predicted value from the model's output
                new_moistures[i] = predictions[0][temp]
                temp += 1
        # print(f'Values after inserting predictions: \n {new_moistures}')

        # print(f'Predicted moisture levels: {new_moistures}')
        actual = [moisture_levels_actual[i] for i in range(len(moisture_levels_actual)) if i not in visited_fields]
        # print(f'Average actual moisture: {sum(actual) / len(actual)}')
        # print(f'Average predicted moisture: {sum(new_moistures) / len(new_moistures)}')
        mse = mean_squared_error(actual, predictions[0])
        mse_vals.append(mse)
        # print(f'Actual moisture levels: \n{actual}')
        # print(f'Predicted moisture levels: \n{predictions[0]}')
        # errors = [actual[i] - predictions[0][i] for i in range(len(actual))]  # Compute the error (actual - predicted)
        # average_error = sum(errors) / len(errors)  # Calculate the average error    
        # mse_vals.append(average_error)
        new_moistures_perc = [x / simulations[0].SAT for x in new_moistures]
        irrigate = [1 if x < threshold else 0 for x in new_moistures_perc]
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
            final_water_storage, final_results, final_water_flux = sim.run_sim(schedule, end)
            q.put((sim.id, final_results, final_water_storage, final_water_flux))
        except Exception as e:
            print(f'Error running simulation {sim.id}: {e}')
            q.put((sim.id, None, None, None))

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
    final_flux = [None] * len(simulations)
    while not result_queue.empty():
        sim_id, results, water_storage, water_flux = result_queue.get()
        final_results[sim_id] = results
        final_storage[sim_id] = water_storage
        final_flux[sim_id] = water_flux

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
    return final_results, final_storage, final_flux, mse_vals


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
    center_sand = random.randint(20, 50)  # Example range for Sand
    center_clay = random.randint(20, 50)  # Example range for Clay
    center_silt = 100 - center_sand - center_clay

    # Initialize the center field (or start from any field)
    sand_values[0, 0] = center_sand
    clay_values[0, 0] = center_clay
    silt_values[0, 0] = center_silt

    # Define a function to generate neighboring values with smooth variations
    def generate_neighbor_values(sand, clay, variation=8):
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
                    sand_values[i, j-1], clay_values[i, j-1]
                )
            elif j == 0:
                # Vertical neighbors only
                sand_values[i, j], clay_values[i, j], silt_values[i, j] = generate_neighbor_values(
                    sand_values[i-1, j], clay_values[i-1, j]
                )
            else:
                # Average neighbors and then add variation
                avg_sand = (sand_values[i-1, j] + sand_values[i, j-1]) // 2
                avg_clay = (clay_values[i-1, j] + clay_values[i, j-1]) // 2
                sand_values[i, j], clay_values[i, j], silt_values[i, j] = generate_neighbor_values(
                    avg_sand, avg_clay
                )

    # Now sand_values, clay_values, and silt_values have smoothly varying values for each grid cell
    soil_values = []
    for i in range(num_rows):
        for j in range(num_cols):
            soil_values.append((sand_values[i, j], clay_values[i, j], silt_values[i, j]))

    return soil_values


def get_total_results(field_length, field_width, final_results):
    total_yield = 0
    total_irr = 0
    field_size_m = field_length * field_width
    field_size_ha = field_size_m / 10000
    for i, results in enumerate(final_results):
        total_yield += results['Dry yield (tonne/ha)'][0] * field_size_ha
        total_irr += results['Seasonal irrigation (mm)'][0] * field_size_m / 1000
    return total_yield, total_irr


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
# def train_model(x_train, y_train):
#     """Train the model using the selected features and target sensors."""
#     # Create pipeline for regression model
#     if len(y_train.columns) == 1:
#         pipeline = Pipeline([
#             ('scaler', StandardScaler()),
#             ('regressor', TransformedTargetRegressor(
#                 regressor=LinearRegression(),
#                 transformer=StandardScaler()
#             ))
#         ])
#     else:
#         pipeline = Pipeline([
#             ('scaler', StandardScaler()),
#             ('regressor', TransformedTargetRegressor(
#                 regressor=MultiOutputRegressor(LinearRegression()),
#                 transformer=StandardScaler()
#             ))
#         ])
#     # Train the model
#     pipeline.fit(x_train, y_train)
    
#     return pipeline

import re

def extract_sensor_values_from_file(file_content: str):
    # Use regex to extract the list of sensors used
    sensor_list_match = re.search(r"sensors used: \[(.*?)\]", file_content)
    
    if sensor_list_match:
        # Extract sensor names and convert them to integers
        sensor_list_str = sensor_list_match.group(1)
        sensor_ids = re.findall(r'sensor_(\d+)', sensor_list_str)
        sensor_ids = [int(sensor_id) for sensor_id in sensor_ids]
        return sensor_ids
    else:
        return None

import os

def get_sensor_values_for_maps(base_folder: str):
    maps = []
    # seeds = [1, 3, 4, 5, 7, 8, 9, 10, 12, 13]
    seeds = [5]
    # Loop through each map folder (map0 to map9)
    for seed in seeds:
        map_folder = os.path.join(base_folder, f"seed{seed}")
        map_data = []
        
        # Loop through each of the files output10k.txt to output50k.txt
        # for output_file in ["output10k.txt", "output15k.txt", "output20k.txt", "output25k.txt", "output30k.txt", "output35k.txt", "output40k.txt", "output45k.txt", "output50k.txt"]:
        # for output_file in ["output15k.txt", "output30k.txt", "output45k.txt", "output50k.txt"]:
        for output_file in ["output40k.txt"]:

            file_path = os.path.join(map_folder, output_file)
            
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    file_content = f.read()
                    sensor_ids = extract_sensor_values_from_file(file_content)
                    
                    # Replace the list with an empty list if it contains 25 elements
                    if sensor_ids and len(sensor_ids) == 25:
                        map_data.append([])
                    else:
                        map_data.append(sensor_ids)
            else:
                map_data.append([])  # If the file does not exist, add an empty list

        maps.append(map_data)
    
    return maps




if __name__ == "__main__":
    # Irrigation comparison for EG seed 1, 15k, 30k, 45k, 50k
    base_folder = "Energy Budget Experiments Soils"
    maps = get_sensor_values_for_maps(base_folder)
    print(maps)
    # budgets = ['15k', '30k', '45k', '50k']
    budgets = ['40k']
    # Initialize lists to store the results of multiple runs
    yields_runs = []
    irrs_runs = []
    mses_runs = []

    seeds = [5]
    # seeds = [1]
    # Perform multiple runs of the experiment
    for i, map_sensors in enumerate(maps):
        seed = seeds[i]
        random.seed(seed)
        np.random.seed(seed)
        soil_values = get_soil_values(5)
        training_data = pd.read_csv(f'training_data0514_seed{seed}.csv', index_col=0)
        yields = []
        irrs = []
        mses = []
        for i, selected_sensors in enumerate(map_sensors):
            if len(selected_sensors) == 0:
                # Do a run without predictions
                year_results, year_storages, year_flux = run_simulations(soil_values=soil_values, start_date=pd.to_datetime('2015/05/01'), end_date='2016/05/01', threshold=0.55)
                for j, flux in enumerate(year_flux):
                    flux['IrrDay'].to_csv(f'Irrigation Comparison/data/irr_seed{seed}_nonPredictive_map{j}_55.csv')
                # total_yield, total_irr = get_total_results(field_length=200, field_width=200, final_results=year_results)
                # yields.append(total_yield)
                # irrs.append(total_irr)
                # mses.append(0)
                continue
            x_train = training_data[[f'sensor_{i}' for i in selected_sensors]]
            y_train = training_data[[f'sensor_{i}' for i in range(25) if i not in selected_sensors]]
            model = train_model(x_train, y_train)
            print('Model trained')
            # Now run the simulation with the trained model
            final_results, final_storages, final_flux, mse_vals = run_simulations_predictive(
                soil_values=soil_values, visited_fields=selected_sensors, model=model, 
                start_date=pd.to_datetime('2015/05/01'), end_date='2016/05/01', threshold=0.55
            )
            # exit()
            # print(final_results)
            # exit()
            # print(final_flux[0]['IrrDay'])
            # with open(f'Irrigation Comparison/irr_seed{seed}_budget{budgets[i]}.txt', 'w') as f:
            #     f.write(str(final_flux[0]['IrrDay']))
            for j, flux in enumerate(final_flux):
                flux['IrrDay'].to_csv(f'Irrigation Comparison/data/irr_seed{seed}_budget{budgets[i]}_map{j}_55.csv')
            # exit()
            # total_yield, total_irr = get_total_results(field_length=200, field_width=200, final_results=final_results)
            # print(f'Total yield: {total_yield} tonnes')
            # print(f'Total irrigation: {total_irr} cubic meters')
            # print(f'Mean MSE: {np.mean(mse_vals)}')
            # yields.append(total_yield)
            # irrs.append(total_irr)
            # mses.append(np.mean(mse_vals))

        exit()
        # Store results from this run
        yields_runs.append(yields)
        irrs_runs.append(irrs)
        mses_runs.append(mses)



    # Convert to numpy arrays for easier calculation
    yields_runs = np.array(yields_runs)
    irrs_runs = np.array(irrs_runs)
    mses_runs = np.array(mses_runs)


    # Calculate the mean and standard deviation for each metric
    yields_mean = np.mean(yields_runs, axis=0)
    irrs_mean = np.mean(irrs_runs, axis=0)
    mses_mean = np.mean(mses_runs, axis=0)

    yields_std = np.std(yields_runs, axis=0)
    irrs_std = np.std(irrs_runs, axis=0)
    mses_std = np.std(mses_runs, axis=0)



    # X-axis labels and range
    x_labels = ['10k', '15k', '20k', '25k', '30k', '35k', '40k', '45k', '50k']
    # x_labels = ['20k', '25k', '30k', '35k', '40k', '45k', '50k']

    # save results to a file
    yields_df = pd.DataFrame(yields_runs, index=seeds, columns=x_labels)
    irrs_df = pd.DataFrame(irrs_runs, index=seeds, columns=x_labels)
    mses_df = pd.DataFrame(mses_runs, index=seeds, columns=x_labels)
        
    # Save results to results.txt file
    # with open('Energy Budget Experiments Soils/results.txt', 'w') as f:
    #     f.write("Yields (tonnes):\n")
    #     f.write(yields_df.to_string())
    #     f.write("\n\nIrrigation (cubic meters):\n")
    #     f.write(irrs_df.to_string())
    #     f.write("\n\nMean Squared Error (MSE):\n")
    #     f.write(mses_df.to_string())





    # Regular Experiments
    base_folder = "Energy Budget Experiments Soils"
    maps = get_sensor_values_for_maps(base_folder)
    print(maps)

    # Initialize lists to store the results of multiple runs
    yields_runs = []
    irrs_runs = []
    mses_runs = []

    seeds = [1, 3, 4, 5, 7, 8, 9, 10, 12, 13]
    # seeds = [1]
    # Perform multiple runs of the experiment
    for i, map_sensors in enumerate(maps):
        seed = seeds[i]
        random.seed(seed)
        np.random.seed(seed)
        soil_values = get_soil_values(5)
        training_data = pd.read_csv(f'training_data0514_seed{seed}.csv', index_col=0)
        yields = []
        irrs = []
        mses = []
        for selected_sensors in map_sensors:
            if len(selected_sensors) == 0:
                # Do a run without predictions
                year_results, year_storages = run_simulations(soil_values=soil_values, start_date=pd.to_datetime('2015/05/01'), end_date='2016/05/01', threshold=0.65)
                total_yield, total_irr = get_total_results(field_length=200, field_width=200, final_results=year_results)
                yields.append(total_yield)
                irrs.append(total_irr)
                mses.append(0)
                continue
            x_train = training_data[[f'sensor_{i}' for i in selected_sensors]]
            y_train = training_data[[f'sensor_{i}' for i in range(25) if i not in selected_sensors]]
            model = train_model(x_train, y_train)
            print('Model trained')
            # Now run the simulation with the trained model
            final_results, final_storages, mse_vals = run_simulations_predictive(
                soil_values=soil_values, visited_fields=selected_sensors, model=model, 
                start_date=pd.to_datetime('2015/05/01'), end_date='2016/05/01', threshold=0.65
            )
            
            total_yield, total_irr = get_total_results(field_length=200, field_width=200, final_results=final_results)
            print(f'Total yield: {total_yield} tonnes')
            print(f'Total irrigation: {total_irr} cubic meters')
            print(f'Mean MSE: {np.mean(mse_vals)}')
            yields.append(total_yield)
            irrs.append(total_irr)
            mses.append(np.mean(mse_vals))

        # Store results from this run
        yields_runs.append(yields)
        irrs_runs.append(irrs)
        mses_runs.append(mses)



    # Convert to numpy arrays for easier calculation
    yields_runs = np.array(yields_runs)
    irrs_runs = np.array(irrs_runs)
    mses_runs = np.array(mses_runs)


    # Calculate the mean and standard deviation for each metric
    yields_mean = np.mean(yields_runs, axis=0)
    irrs_mean = np.mean(irrs_runs, axis=0)
    mses_mean = np.mean(mses_runs, axis=0)

    yields_std = np.std(yields_runs, axis=0)
    irrs_std = np.std(irrs_runs, axis=0)
    mses_std = np.std(mses_runs, axis=0)



    # X-axis labels and range
    x_labels = ['10k', '15k', '20k', '25k', '30k', '35k', '40k', '45k', '50k']
    # x_labels = ['20k', '25k', '30k', '35k', '40k', '45k', '50k']

    # save results to a file
    yields_df = pd.DataFrame(yields_runs, index=seeds, columns=x_labels)
    irrs_df = pd.DataFrame(irrs_runs, index=seeds, columns=x_labels)
    mses_df = pd.DataFrame(mses_runs, index=seeds, columns=x_labels)
        
    # Save results to results.txt file
    # with open('Energy Budget Experiments Soils/results.txt', 'w') as f:
    #     f.write("Yields (tonnes):\n")
    #     f.write(yields_df.to_string())
    #     f.write("\n\nIrrigation (cubic meters):\n")
    #     f.write(irrs_df.to_string())
    #     f.write("\n\nMean Squared Error (MSE):\n")
    #     f.write(mses_df.to_string())

    x_range = np.arange(len(x_labels))

    # Create a figure with 3 subplots (one row, three columns)
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    # Plot the total yields with error bars
    axs[0].errorbar(x_range, yields_mean, yerr=yields_std, label='Total Yield (tonnes)', marker='o', color='b', capsize=5)
    axs[0].set_ylabel('Yield (tonnes)')
    axs[0].set_title('Total Yield vs Energy Budget')
    # axs[0].set_ylim(0, 1500)
    axs[0].grid(True)

    # Plot the total irrigation with error bars
    axs[1].errorbar(x_range, irrs_mean, yerr=irrs_std, label='Total Irrigation (m³)', marker='o', color='g', capsize=5)
    axs[1].set_ylabel('Irrigation (m³)')
    axs[1].set_title('Total Irrigation vs Energy Budget')
    # axs[1].set_ylim(0, 350000)
    axs[1].grid(True)

    # Plot the MSE values with error bars
    axs[2].errorbar(x_range, mses_mean, yerr=mses_std, label='Mean Squared Error (MSE)', marker='o', color='r', capsize=5)
    axs[2].set_ylabel('MSE')
    axs[2].set_xlabel('Energy Budget')
    axs[2].set_title('Mean Squared Error (MSE) vs Energy Budget')
    axs[2].grid(True)

    # Customize x-axis with labels for all subplots
    axs[2].set_xticks(x_range)
    axs[2].set_xticklabels(x_labels)

        
    # Calculate the average number of sensors per position, counting empty lists as 25
    average_num_sensors = []
    sensors_std = []

    # Loop over each index position across sublists
    for idx in range(9):
        num_sensors = []
        for map_data in maps:
            # Check if the current map has a sublist at the position idx
            if idx < len(map_data):
                # Count items in sublist, using 25 if it is empty
                num_sensors.append(len(map_data[idx]) if map_data[idx] else 25)
            else:
                # If sublist does not exist in this map, assume it's empty (count as 25)
                num_sensors.append(25)
        # Calculate mean and standard deviation for this index position
        average_num_sensors.append(np.mean(num_sensors))
        sensors_std.append(np.std(num_sensors))

    # Plot the average number of sensors with error bars on axs[3]
    axs[3].errorbar(x_range, average_num_sensors, yerr=sensors_std, label='Average Number of Sensors', marker='s', color='purple', capsize=5)
    axs[3].set_ylabel('Average Number of Sensors')
    axs[3].set_xlabel('Energy Budget')
    axs[3].set_title('Average Number of Sensors Selected for Each Energy Budget')
    axs[3].grid(True)


    # Create a figure with 3 subplots (one row, three columns)
    fig2, axs2 = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    # Plot the total yields with error bars
    axs2[0].errorbar(x_range, yields_mean, yerr=yields_std, label='Total Yield (tonnes)', marker='o', color='b', capsize=5)
    axs2[0].set_ylabel('Yield (tonnes)')
    axs2[0].set_title('Total Yield vs Energy Budget')
    axs2[0].set_ylim(0, 1500)
    axs2[0].grid(True)

    # Plot the total irrigation with error bars
    axs2[1].errorbar(x_range, irrs_mean, yerr=irrs_std, label='Total Irrigation (m³)', marker='o', color='g', capsize=5)
    axs2[1].set_ylabel('Irrigation (m³)')
    axs2[1].set_title('Total Irrigation vs Energy Budget')
    axs2[1].set_ylim(0, 400000)
    axs2[1].grid(True)

    # Plot the MSE values with error bars
    axs2[2].errorbar(x_range, mses_mean, yerr=mses_std, label='Mean Squared Error (MSE)', marker='o', color='r', capsize=5)
    axs2[2].set_ylabel('MSE')
    axs2[2].set_xlabel('Energy Budget')
    axs2[2].set_title('Mean Squared Error (MSE) vs Energy Budget')
    axs2[2].grid(True)

    # Customize x-axis with labels for all subplots
    axs2[2].set_xticks(x_range)
    axs2[2].set_xticklabels(x_labels)

    axs2[3].errorbar(x_range, average_num_sensors, yerr=sensors_std, label='Average Number of Sensors', marker='s', color='purple', capsize=5)
    axs2[3].set_ylabel('Average Number of Sensors')
    axs2[3].set_xlabel('Energy Budget')
    axs2[3].set_title('Average Number of Sensors Selected for Each Energy Budget')
    axs2[3].grid(True)

    # Adjust layout to avoid overlap and display plot
    plt.tight_layout()
    plt.show()
