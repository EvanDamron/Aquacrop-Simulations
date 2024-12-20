from shapely import Point
from shapely.geometry import Polygon
from mapping import plotPath, getSensorNames, addSensorsUniformRandom, minSetCover
import matplotlib.pyplot as plt
from python_tsp.distances import euclidean_distance_matrix
from python_tsp.heuristics import solve_tsp_local_search
from ML import getMSE, calculate_feature_importance, normalizeData, processData
import random
import numpy as np
from algorithmHelpers import processFiles, getEnergy, remBestHP, createMSEPlot, updateMSEPlot, printResults, \
    writeResults
import time
import geopandas as gpd
import pandas as pd
import math



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



def hoverPointsTSP(points, scramble=False):
    # print(f'points: {points}')
    if len(points) == 0:
        gdf = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in points], crs='EPSG:3857')
        return gdf, 0
    depotLocation = (0, 0)
    points = np.insert(points, 0, depotLocation, axis=0)
    distanceMatrix = euclidean_distance_matrix(points)
    if scramble == False:
        initPerm = list(range(len(points)))
        permutation, distance = solve_tsp_local_search(distance_matrix=distanceMatrix, x0=initPerm)
    else:
        permutation, distance = solve_tsp_local_search(distance_matrix=distanceMatrix)
    orderedPoints = [points[i] for i in permutation if i != 0]
    gdf = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in orderedPoints], crs='EPSG:3857')
    return gdf, distance




# fig1, ax1, HP_gdf, sensorNames, df,
def RSEO(fig1, ax1, HP_gdf, sensorNames, df,
         savePlots=False, pathPlotName="", msePlotName="", outputTextName="", droneHeight=15,
         communicationRadius=70, energyBudget=60000, joulesPerMeter=10, addToOriginal=True,
         joulesPerSecond=35, dataSize=100, transferRate=9, minSet=True, generate=False, numSensors=37):
    startTime = time.time()
    # fig1, ax1, HP_gdf, selected, unselected, sensorNames, df = processFiles(droneHeight, communicationRadius, minSet,
    #                                                                         generate, addToOriginal, numSensors)

    # print(newDF)
    for i in range(5):
        sensorNames, HP_gdf = minSetCover(sensorNames, HP_gdf)
    selected = HP_gdf.copy()
    HP_gdf.plot(ax=ax1, color='yellow', markersize=10, alpha=1)
    sensorImportances = calculate_feature_importance(df)
    hoverPointImportances = dict.fromkeys(sensorNames.keys(), 0)
    for hoverPoint in HP_gdf['geometry']:
        for sensor in sensorNames[hoverPoint]:
            hoverPointImportances[hoverPoint] += sensorImportances[sensor]
    x, y, line, ax2, fig2 = createMSEPlot()
    selected, distance = getEnergy(selected, sensorNames, joulesPerMeter, joulesPerSecond, dataSize, transferRate)
    loopCount = 0
    while selected['energy'][0] > energyBudget:
        loopCount += 1
        lowestScore = min(hoverPointImportances[point] for point in selected['geometry'])
        lowestScorePoints = [key for key, value in hoverPointImportances.items() if value == lowestScore and key in selected['geometry']]
        # if len(lowestScorePoints) > 1:
        #
        #     print('points were tied...')
        #     exit(1)
        lowestScorePoint = lowestScorePoints[0]
        selected = selected[selected['geometry'] != lowestScorePoint]
        selected, distance = getEnergy(selected, sensorNames, joulesPerMeter, joulesPerSecond, dataSize, transferRate)
        # features = getSensorNames(selected['geometry'], sensorNames)
        # mse = getMSE(features, df)
        # mse = 1
        # updateMSEPlot(loopCount, mse, ax2, fig2, x, y, line)
        if len(selected) == 0:
            break
        # print(f"MSE: {mse}")
    # if len(selected) == 0:
    #     print('no solution found')
    #     return 0
    features = getSensorNames(selected['geometry'], sensorNames)
    print(f'features {features}, {len(features)}')
    targets = [name for name in df.columns if name not in features]
    print(f'targets {targets}, {len(targets)}')
    mse = getMSE(features, df)
    printResults(finalSHP=selected, finalDistance=distance, finalIteration=loopCount, finalMSE=mse,
                 sensorNames=sensorNames)
    if len(selected) != 0:
        plotPath(ax1, selected)
    for key, value in sensorNames.items():
            ax1.text(key.x, key.y, str(value), fontsize=10, ha='center', va='bottom')
    if savePlots:
        writeResults(selected, loopCount, distance, mse, sensorNames, outputTextName, startTime, finalMSE2=0)
        fig1.savefig(pathPlotName, bbox_inches='tight')
        fig2.savefig(msePlotName, bbox_inches='tight')
        return mse
    else:
        print(df)
        print(hoverPointImportances)
        plt.show()


if __name__ == '__main__':
    # training_data = pd.read_csv('training_data0514.csv', index_col=0)
    soil_seeds = [3, 4, 5, 7, 8, 9, 10, 12, 13]
    # soil_seeds = [1, 3]
    # data_paths = ['training_data0514_seed3.csv', 'training_data0514_seed4.csv', 'training_data0514_seed5.csv', 'training_data0514_seed7.csv']
    # print(training_data)[;[]pl,[]]
    # plt.show()
    energy_budgets = [10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000]
    for seed in soil_seeds:
        training_data = pd.read_csv(f'training_data0514_seed{seed}.csv', index_col=0)
        for budget in energy_budgets:
            np.random.seed(0)
            random.seed(0)
            budget_str = f'{budget//1000}k'
            fig, ax, hoverPoints, sensorNames, sensor_gdf = create_map(15, 70, training_data.columns)
            RSEO(fig1=fig, ax1=ax, HP_gdf=hoverPoints, sensorNames=sensorNames, df=training_data, energyBudget=budget, savePlots=True, 
                 pathPlotName=f'Energy Budget Experiments Soils RSEO/seed{seed}/pathPlot{budget_str}.png', 
                 msePlotName=f'Energy Budget Experiments Soils RSEO/seed{seed}/msePlot{budget_str}.png', 
                 outputTextName=f'Energy Budget Experiments Soils RSEO/seed{seed}/output{budget_str}.txt')
            