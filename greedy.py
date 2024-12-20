import math
from mapping import findMinTravelDistance, plotPath, getSensorNames, addSensorsUniformRandom, getHoverPoints
import pandas as pd
import matplotlib.pyplot as plt
from ML import getMSE, calculateEntropy, discretizeData, getConditionalEntropy, getInformationGain, processData,normalizeData
import random
import time
import numpy as np
from algorithmHelpers import processFiles, getEnergy, getPointsInBudget, \
    createMSEPlot, updateMSEPlot, printResults, writeResults, printTime, createIGMSEPlot, updateIGMSEPlot, \
    addBestHybrid, remBestHybrid
import signal
import sys
import geopandas as gpd
from joblib import Parallel, delayed
from shapely.geometry import Polygon, Point
from sklearn.preprocessing import StandardScaler



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


def getNonRedundantPointsInBudget(unselected, selected, sensorNames, energyBudget, joulesPerMeter, joulesPerSecond, dataSize,
                      transferRate):
    inBudgetHP = unselected.copy()

    def calculateEnergyIfAdded(index, row):
        extractedRow = gpd.GeoDataFrame([row], geometry='geometry', crs=unselected.crs)
        tempSHP = gpd.GeoDataFrame(pd.concat([selected, extractedRow], ignore_index=True), crs=unselected.crs)
        tempSHP, _ = getEnergy(tempSHP, sensorNames, joulesPerMeter, joulesPerSecond, dataSize, transferRate)
        return index, tempSHP['energy'][0]

    indexesEnergy = Parallel(n_jobs=-1)(
        delayed(calculateEnergyIfAdded)(index, row) for index, row in inBudgetHP.iterrows())
    indexesEnergy.sort(key=lambda x: x[0])
    for value in indexesEnergy:
        index, energy = value
        if energy > energyBudget:
            inBudgetHP = inBudgetHP.drop(index)
        else:
            inBudgetHP.at[index, 'energyIfAdded'] = energy
    if 'energy' in inBudgetHP.columns:
        inBudgetHP.drop('energy', axis=1, inplace=True)
    inBudgetHP = inBudgetHP.reset_index(drop=True)

    # Remove redundant points
    # inBudgetNonRedundant = inBudgetHP.copy()
    # for point in inBudgetHP['geometry']:
    #     if all(sensor in getSensorNames(selected['geometry'], sensorNames) for sensor in sensorNames[point]):
    #         inBudgetNonRedundant = inBudgetNonRedundant[inBudgetNonRedundant['geometry'] != point]
    inBudgetNonRedundant = inBudgetHP.copy()
    nonRedundantPoints = []

    for point in inBudgetHP['geometry']:
        # Get the sensors currently covered by the selected points
        coveredSensors = getSensorNames(selected['geometry'], sensorNames)

        # Check if all sensors of the current point are already covered
        # if not all(sensor in coveredSensors for sensor in sensorNames[point]):
        #     nonRedundantPoints.append(point)
        if not any(sensor in coveredSensors for sensor in sensorNames[point]):
            nonRedundantPoints.append(point)

    # Filter the DataFrame to keep only non-redundant points
    inBudgetNonRedundant = inBudgetNonRedundant[inBudgetNonRedundant['geometry'].isin(nonRedundantPoints)]
    return inBudgetNonRedundant.reset_index(drop=True)



def addBestHP(unselected, unselectedIB, selected, sensorNames, df, energyWeight):
    print("ADD BEST")
    rewards = [-10] * len(unselectedIB)
    if unselectedIB.empty:
        print('No more hover points within budget')
        return unselected, selected
    if selected.empty:  # never happens in actual greedy alg
        oldEnergy = 0
        oldIG = float('-inf')
    else:
        oldEnergy = selected['energy'][0]
        oldFeatures = getSensorNames(selected['geometry'], sensorNames)
        oldIG = getInformationGain(oldFeatures, df)


    def calculateIGAdding(index, row):
        extractedRow = gpd.GeoDataFrame([row], geometry='geometry', crs=unselectedIB.crs)
        tempSHP = gpd.GeoDataFrame(pd.concat([selected, extractedRow], ignore_index=True), crs=unselectedIB.crs)
        features = getSensorNames(tempSHP['geometry'], sensorNames)
        informationGain = getInformationGain(features, df)  # for minimizing entropy
        newEnergy = unselectedIB['energyIfAdded'][index]
        return index, informationGain, newEnergy



    indexIGEnergy = Parallel(n_jobs=-1)(delayed(calculateIGAdding)
                                     (index, row) for index, row in unselectedIB.iterrows())
    indexIGEnergy = sorted(indexIGEnergy, key=lambda x: x[0])

    # Proportion scaling
    # total_IG = sum(IG for _, IG, _ in indexIGEnergy)
    # total_energy = sum(newEnergy for _, _, newEnergy in indexIGEnergy)
    #
    # for index, IG, newEnergy in indexIGEnergy:
    #     reward_IG_ratio = IG / total_IG if total_IG != 0 else 0
    #     reward_energy_ratio = newEnergy / total_energy if total_energy != 0 else 1
    #     weighted_energy = reward_energy_ratio * energyWeight
    #     weighted_IG = reward_IG_ratio * (1 - energyWeight)
    #     rewards[index] = weighted_IG - weighted_energy

    # Standard Scaler

    # Extract information gain (IG) and energy values from the data for standardization
    ig_values = np.array([IG for _, IG, _ in indexIGEnergy]).reshape(-1, 1)
    energy_values = np.array([newEnergy for _, _, newEnergy in indexIGEnergy]).reshape(-1, 1)

    # Apply StandardScaler to IG and energy values
    scaler = StandardScaler()

    ig_values_scaled = scaler.fit_transform(ig_values).flatten()
    energy_values_scaled = scaler.fit_transform(energy_values).flatten()

    for i, (index, IG, newEnergy) in enumerate(indexIGEnergy):
        # Use the standardized IG and energy values
        standardizedIG = ig_values_scaled[i]
        standardizedEnergy = energy_values_scaled[i]

        # Compute the weighted values for energy and IG
        weighted_energy = standardizedEnergy * energyWeight
        weighted_IG = standardizedIG * (1 - energyWeight)

        # Calculate the final reward value
        rewards[index] = weighted_IG - weighted_energy


    # Min-Max for IG, Proportion for energy
    # minIG = min(IG for _, IG, _ in indexIGEnergy)
    # maxIG = max(IG for _, IG, _ in indexIGEnergy)
    # total_energy = sum(newEnergy for _, _, newEnergy in indexIGEnergy)
    # for index, IG, newEnergy in indexIGEnergy:
    #     normalizedIG = (IG - minIG) / (maxIG - minIG) if maxIG != minIG else 0
    #     reward_energy_ratio = newEnergy / total_energy if total_energy != 0 else 0
    #     # print(f"IG - EG: {normalizedIG} - {reward_energy_ratio} = {normalizedIG - reward_energy_ratio}")
    #     weighted_energy = reward_energy_ratio * energyWeight
    #     weighted_IG = normalizedIG * (1 - energyWeight)
    #     # print(f"weighted: {weighted_IG} - {weighted_energy} = {weighted_IG - weighted_energy}")
    #     rewards[index] = weighted_IG - weighted_energy


    # Min-Max scaling
    # minIG = min(IG for _, IG, _ in indexIGEnergy)
    # maxIG = max(IG for _, IG, _ in indexIGEnergy)
    # minEnergy = min(newEnergy for _, _, newEnergy in indexIGEnergy)
    # maxEnergy = max(newEnergy for _, _, newEnergy in indexIGEnergy)
    # for index, IG, newEnergy in indexIGEnergy:
    #     normalizedIG = (IG - minIG) / (maxIG - minIG) if maxIG != minIG else 0
    #     normalizedEnergy = (newEnergy - minEnergy) / (maxEnergy - minEnergy) if maxEnergy != minEnergy else 0
    #     weighted_energy = normalizedEnergy * energyWeight
    #     reward = normalizedIG / (0.000001 + weighted_energy)
    #     rewards[index] = reward
    # Take best IG
    # for index, IG, newEnergy in indexIGEnergy:
    #     rewards[index] = IG

    # make sure not to add a redundant point
    for index, IG, newEnergy in indexIGEnergy:
        if IG == oldIG:
            # print('IG tie')
            # print(f"old IG: {oldIG}, new IG: {IG} Energy: {newEnergy}")
            # print(f"{selected['geometry'][index]} corresponds to {getSensorNames([selected['geometry'][index]], sensorNames)}")
            # print(f"selected sensors: {getSensorNames(selected['geometry'], sensorNames)}")
            rewards[index] = -999
        # print(f"IG: {IG}, Energy: {newEnergy}, Reward: {rewards[index]}")
    maxReward = max(rewards)
    indexesEnergiesOfBest = []
    # get all the points resulting in the same mse
    for index, IG, newEnergy in indexIGEnergy:
        if rewards[index] == maxReward:
            indexesEnergiesOfBest.append((index, newEnergy))
    if len(indexesEnergiesOfBest) == 1:
        indexOfBest, energyOfBest = indexesEnergiesOfBest[0]
    # if there is a tie, take the one that adds the sensors correlated and remove redundant sensors from selected
    else:
        points = []
        for index, energy in indexesEnergiesOfBest:
            point = unselectedIB.loc[index, 'geometry']
            points.append(point)
        tiedPointSensors = getSensorNames(points, sensorNames)
        numTiedSensors = len(tiedPointSensors)
        if numTiedSensors == 1:
            indexOfBest = min(indexesEnergiesOfBest, key=lambda x: x[1])[0]
        else:
            print('REDUNDANCY CHECK')
            numSensors = -1
            pointOfInterest = None
            for point in points:
                if len(sensorNames[point]) > numSensors:
                    numSensors = len(sensorNames[point])
                    pointOfInterest = point
            uniqueOldPoints = set()
            for pointInSelected in selected['geometry']:
                for sensor in sensorNames[pointInSelected]:
                    if sensor in sensorNames[pointOfInterest]:
                        uniqueOldPoints.add(pointInSelected)
            oldPoints = list(uniqueOldPoints)
            sensorsInOldPoints = set(getSensorNames(oldPoints, sensorNames))
            sensorsInPointOfInterest = set(sensorNames[pointOfInterest])
            if sensorsInPointOfInterest != sensorsInOldPoints.union(sensorsInPointOfInterest):
                indexOfBest = min(indexesEnergiesOfBest, key=lambda x: x[1])[0]
            else:
                updatedSelected = selected[~selected['geometry'].isin(oldPoints)].copy()
                updatedUnselected = unselected[unselected['geometry'] != pointOfInterest].copy()
                newSelectedRow = {'geometry': [pointOfInterest], 'energy': [oldEnergy]}
                newSelectedGDF = gpd.GeoDataFrame(newSelectedRow, geometry='geometry', crs=selected.crs)
                updatedSelected = gpd.GeoDataFrame(pd.concat([updatedSelected, newSelectedGDF], ignore_index=True),
                                                   crs=selected.crs)
                for point in oldPoints:
                    newRow = gpd.GeoDataFrame({'geometry': [point]}, geometry='geometry', crs=selected.crs)
                    updatedUnselected = gpd.GeoDataFrame(pd.concat([updatedUnselected, newRow], ignore_index=True),
                                                         crs=selected.crs)
                return updatedUnselected, updatedSelected

    print('Normal add')
    rowToMove = unselectedIB.loc[[indexOfBest]].copy()
    rowToMove.rename(columns={'energyIfAdded': 'energy'}, inplace=True)
    rowToMove = rowToMove.reset_index(drop=True)
    selected['energy'] = rowToMove['energy'][0]
    rowToMovePoint = rowToMove['geometry'].iloc[0]
    unselected = unselected[unselected['geometry'] != rowToMovePoint]
    selected = gpd.GeoDataFrame(pd.concat([selected, rowToMove], ignore_index=True), crs=unselectedIB.crs)
    return unselected, selected

def greedy_add(fig1, ax1, HP_gdf, sensorNames, df,
                savePlots=False, pathPlotName="", msePlotName="", outputTextName="",
                droneHeight=15, energyWeight=0.0, communicationRadius=70, energyBudget=40000, joulesPerMeter=10,
                joulesPerSecond=35, dataSize=100, transferRate=9, minSet=False, generate=False, numSensors=37,
                addToOriginal=True, exhaustive=False):
    startTime = time.time()
    UHP_gdf = HP_gdf.copy()
    SHP_gdf = HP_gdf.iloc[0:0].copy()
    originalDF = df.copy()
    discreteDF = originalDF.copy()
    # discreteDF = discretizeData(originalDF, numBins=8)
    # x, y1, y2, line1, line2, ax2, ax3, fig2 = createIGMSEPlot()
    print(f"Total number of Hoverpoints: {len(HP_gdf)}")
    print(f"Total number of sensors: {len(getSensorNames(HP_gdf['geometry'], sensorNames))}")
    print(f"GREEDY-ADD ALGORITHM WITH ENERGY BUDGET {energyBudget}")
    pointsInBudget = getNonRedundantPointsInBudget(UHP_gdf, SHP_gdf, sensorNames, energyBudget, joulesPerMeter, joulesPerSecond,
                                       dataSize, transferRate)
    MSEs = []
    IGs = []
    iteration = 0
    while len(pointsInBudget) > 0:
        iteration += 1
        print(f"Total number of points in budget: {len(pointsInBudget)}")
        UHP_gdf, SHP_gdf = addBestHP(unselected=UHP_gdf, unselectedIB=pointsInBudget, selected=SHP_gdf,
                                     sensorNames=sensorNames, df=discreteDF, energyWeight=energyWeight)
        pointsInBudget = getNonRedundantPointsInBudget(UHP_gdf, SHP_gdf, sensorNames, energyBudget, joulesPerMeter,
                                                       joulesPerSecond, dataSize, transferRate)

        print(f"Total number of selected Hoverpoints: {len(SHP_gdf)}")
        features = getSensorNames(SHP_gdf['geometry'], sensorNames)
        print(f"Total number of selected sensors: {len(features)}")
        # mse = getMSE(features, originalDF)
        # mses = []
        # for seed in range(10):
        #     mses.append(getMSE(features, originalDF, seed))
        # mse = np.mean(mses)
        # MSEs.append(mse)
        # IG = getInformationGain(features, discreteDF)
        # IGs.append(IG)
        # print(f"Iteration: {iteration}, MSE: {mse}, IG: {IG}")
        # updateIGMSEPlot(newX=iteration, newY1=mse, newY2=IG,
        #                 line1=line1, line2=line2, ax2=ax2, ax3=ax3,
        #                 fig=fig2, x=x, y1=y1, y2=y2)
    features = getSensorNames(SHP_gdf['geometry'], sensorNames)
    # mses = []
    # for seed in range(10):
    #     mses.append(getMSE(features, originalDF, seed))
    # mse = np.mean(mses)
    # mse_std_dev = np.std(mses)
    mse = getMSE(features, originalDF)
    # IG = getInformationGain(features, discreteDF)
    SHP_gdf, distance = getEnergy(SHP_gdf, sensorNames, joulesPerMeter, joulesPerSecond, dataSize, transferRate)
    printResults(SHP_gdf, iteration, distance, mse, sensorNames)
    plotPath(ax1, SHP_gdf)
    SHP_gdf.plot(ax=ax1, color='red', markersize=10, alpha=1)
    # plt.show()
    # return mse, len(SHP_gdf)
    if savePlots:
        writeResults(SHP_gdf, iteration, distance, mse, sensorNames, outputTextName, startTime, mse)
        fig1.savefig(pathPlotName, bbox_inches='tight')
        # fig2.savefig(msePlotName, bbox_inches='tight')
    else:
        plt.show()


if __name__ == '__main__':
    soil_seeds = [1, 3, 4, 5, 7, 8, 9, 10, 12, 13]
    energy_budgets = [10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000]
    # energy_budgets = [10000, 15000]
    for seed in soil_seeds:
        training_data = pd.read_csv(f'Training Data/training_data0514_seed{seed}.csv', index_col=0)
        for budget in energy_budgets:
            np.random.seed(0)
            random.seed(0)
            budget_str = f'{budget//1000}k'
            fig, ax, hoverPoints, sensorNames, sensor_gdf = create_map(15, 70, training_data.columns)
            greedy_add(fig1=fig, ax1=ax, HP_gdf=hoverPoints, sensorNames=sensorNames, df=training_data,
                        savePlots=True, pathPlotName=f"Energy Budget Experiments Soils Greedy/seed{seed}/pathPlot{budget_str}.png",
                        outputTextName=f"Energy Budget Experiments Soils Greedy/seed{seed}/output{budget_str}.txt",
                        energyBudget=budget, energyWeight=0.1)








    droneHeight = 15
    communicationRadius = 70
    minSet = False
    generate = False
    addToOriginal = True
    numSensors = 37
    processed_files = processFiles(droneHeight, communicationRadius, minSet, generate, addToOriginal, numSensors)
    # for i in range(2):
    #     processed_files = processFiles(droneHeight, communicationRadius, minSet, generate, addToOriginal, numSensors)
    #     print(f"Experiment {i}")
    #     greedy_add(processed_files, savePlots=True, pathPlotName=f"Greedy Experiments/greedy_add_path{i}.png",
    #                msePlotName=f"Greedy Experiments/greedy_add_mse{i}.png",
    #                outputTextName=f"Greedy Experiments/greedy_add_output{i}.txt")
    #     print(f"Experiment {i} done")
    # gamma_values = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1]
    # gamma_values = [0]
    energy_budgets = [20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000]
    # energy_budgets = [55000]
    for energyBudget in energy_budgets:
        processed_files = processFiles(droneHeight, communicationRadius, minSet, generate, addToOriginal, numSensors)
        energyBudget_str = f'{energyBudget // 1000}k'
        greedy_add(processed_files, savePlots=True,
                   pathPlotName=f"Greedy Experiments/greedy_add_path{energyBudget_str}.png",
                   msePlotName=f"Greedy Experiments/greedy_add_mse{energyBudget_str}.png",
                   outputTextName=f"Greedy Experiments/greedy_add_output{energyBudget_str}.txt",
                   energyBudget=energyBudget, energyWeight=0.1)
    exit()
    all_mses = {gamma: [] for gamma in gamma_values}
    all_num_sensors = {gamma: [] for gamma in gamma_values}
    for gamma in gamma_values:
        for energyBudget in energy_budgets:
            processed_files = processFiles(droneHeight, communicationRadius, minSet, generate, addToOriginal, numSensors)
            print(f"Experiment with gamma {gamma} and energy budget {energyBudget}")
            # energyBudget_str = f'{energyBudget // 1000}k'
            # greedy_add(processed_files, savePlots=True,
            #            pathPlotName=f"Greedy Experiments/greedy_add_path{energyBudget_str}.png",
            #            msePlotName=f"Greedy Experiments/greedy_add_mse{energyBudget_str}.png",
            #            outputTextName=f"Greedy Experiments/greedy_add_output{energyBudget_str}.txt",
            #            energyBudget=energyBudget, energyWeight=i / energyBudget)
            mse, num_sensors = greedy_add(processed_files, savePlots=False,
                               energyBudget=energyBudget, energyWeight=gamma)
            all_mses[gamma].append(mse)
            all_num_sensors[gamma].append(num_sensors)

    # Plotting MSE over Energy Budgets for each Gamma value
    plt.figure(figsize=(10, 5))
    for gamma in gamma_values:
        plt.plot(energy_budgets, all_mses[gamma], label=f'Gamma={gamma}')
    plt.xlabel('Energy Budget')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('MSE vs Energy Budgets for Different Gamma Values, Proportion Scaling')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotting Number of Sensors over Energy Budgets for each Gamma value
    plt.figure(figsize=(10, 5))
    for gamma in gamma_values:
        plt.plot(energy_budgets, all_num_sensors[gamma], label=f'Gamma={gamma}')
    plt.xlabel('Energy Budget')
    plt.ylabel('Number of Sensors')
    plt.title('Number of Sensors vs Energy Budgets for Different Gamma Values, Proportion Scaling')
    plt.legend()
    plt.grid(True)
    plt.show()