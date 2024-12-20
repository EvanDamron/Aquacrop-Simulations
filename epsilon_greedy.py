import time
import numpy as np
#from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.distances import euclidean_distance_matrix
from python_tsp.heuristics import solve_tsp_local_search
import geopandas as gpd
from shapely import Point
from algorithmHelpers import processFiles, getEnergy, getPointsInBudget, addRandomHP, remRandomHP, addBestHP, remBestHP, \
    createMSEPlot, updateMSEPlot, printResults, writeResults, printTime, createIGMSEPlot, updateIGMSEPlot
from mapping import findMinTravelDistance, plotPath, getSensorNames, addSensorsUniformRandom, getHoverPoints
from ML import getMSE, calculateEntropy, discretizeData, getConditionalEntropy, getInformationGain, processData,normalizeData
import random
import pandas as pd
import math
from matplotlib import pyplot as plt
from shapely import Point, Polygon
import sys
import signal



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


def signal_handler(sig, frame):
    print('Ctrl+C pressed! Showing plot...')
    plt.show()  # display the plot
    sys.exit(0)  # exit the program



def epsilonGreedy(fig1, ax1, HP_gdf, sensorNames, df, numLoops, startTime,
                  savePlots=False, pathPlotName="", msePlotName="", outputTextName="",
                  energyWeight=0, energyBudget=40000, joulesPerMeter=10,
                  joulesPerSecond=35, dataSize=100, transferRate=9, exhaustive=True):
    UHP_gdf = HP_gdf.copy()
    SHP_gdf = HP_gdf.iloc[0:0].copy()

    originalDF = df.copy()
    if not exhaustive:
        df = discretizeData(df)
    # discreteDF = None
    signal.signal(signal.SIGINT, signal_handler)
    pd.set_option('display.max_rows', 30)
    pd.set_option('display.max_columns', None)
    # print(discreteDF)
    # print(originalDF)
    x, y1, y2, line1, line2, ax2, ax3, fig2 = createIGMSEPlot()

    print(f"Total number of Hoverpoints: {len(HP_gdf)}")
    print(f"Total number of sensors: {len(getSensorNames(HP_gdf['geometry'], sensorNames))}")
    print(f"EPSILON-GREEDY ALGORITHM WITH {numLoops} LOOPS AND ENERGY BUDGET {energyBudget}")
    arProb = 0  # probability of adding (0) and removing (1)
    rbProb = 1  # probability of random (1) and best (0)
    loopCount = 0
    pointsInBudget = getPointsInBudget(UHP_gdf, SHP_gdf, sensorNames, energyBudget, joulesPerMeter, joulesPerSecond,
                                       dataSize, transferRate)

    minMSE = 1000
    maxIG = float('-inf')
    # Make sure every hover-point in hp is accounted for in uhp and shp
    if len(UHP_gdf) + len(SHP_gdf) != len(HP_gdf):
        print('ERROR: SELECTED + UNSELECTED != HP')
        exit(1)
    while loopCount < numLoops:
        loopCount += 1
        # energyWeight = 1 - (loopCount / numLoops)
        print("Loop iteration ", loopCount, ' of ', numLoops)
        raProb = rbProb * (1 - arProb)  # random add
        rrProb = rbProb * arProb  # random remove
        baProb = (1 - rbProb) * (1 - arProb)  # best add
        brProb = (1 - rbProb) * arProb  # best remove
        randomNumber = (random.random())
        if randomNumber < raProb:
            UHP_gdf, SHP_gdf = addRandomHP(UHP_gdf, pointsInBudget, SHP_gdf)
        elif randomNumber < raProb + rrProb:
            UHP_gdf, SHP_gdf = remRandomHP(UHP_gdf, SHP_gdf)
        elif randomNumber < raProb + rrProb + baProb:
            UHP_gdf, SHP_gdf = addBestHP(unselected=UHP_gdf, unselectedIB=pointsInBudget, selected=SHP_gdf,
                                         sensorNames=sensorNames, df=df, energyWeight=energyWeight,
                                         exhaustive=exhaustive)
        else:
            UHP_gdf, SHP_gdf = remBestHP(UHP_gdf, SHP_gdf, sensorNames=sensorNames, df=df, energyWeight=energyWeight,
                                         joulesPerMeter=joulesPerMeter, joulesPerSecond=joulesPerSecond,
                                         dataSize=dataSize, transferRate=transferRate, exhaustive=exhaustive)

        UHP_gdf = UHP_gdf.reset_index(drop=True)
        SHP_gdf = SHP_gdf.reset_index(drop=True)
        rbProb = rbProb - 1 / numLoops

        features = getSensorNames(SHP_gdf['geometry'], sensorNames)
        if exhaustive:
            mse = getMSE(features, originalDF)
            IG = 1
            if mse < minMSE:
                minMSE = mse
                bestSHP = SHP_gdf.copy()
                iterationOfBest = loopCount
        else:
            # mse = getMSE(features, originalDF)
            mse = 1
            IG = getInformationGain(features, df)
            if IG > maxIG:
                maxIG = IG
                bestSHP = SHP_gdf.copy()
                iterationOfBest = loopCount

        if SHP_gdf.empty:
            totalEnergy = 0
        else:
            SHP_gdf, _ = getEnergy(dataSize=dataSize, joulesPerMeter=joulesPerMeter, joulesPerSecond=joulesPerSecond,
                                   selected=SHP_gdf, sensorNames=sensorNames, transferRate=transferRate)
            totalEnergy = SHP_gdf['energy'][0]

        print(f"This set of hoverpoints requires {totalEnergy} Joules out of the"
              f" {energyBudget} Joules in the drone's battery")
        print('Total Number of Hover Points Visited: ', len(SHP_gdf))
        print(f"Total number of sensors visited: {len(getSensorNames(SHP_gdf['geometry'], sensorNames))}")
        if exhaustive:
            print(f"current mse: {mse}, lowest mse yet: {minMSE}")
        else:
            print(f"Current Information Gain: {IG}, largest IG yet: {maxIG}")
        
        updateIGMSEPlot(newX=loopCount, newY1=mse, newY2=IG,
                        line1=line1, line2=line2, ax2=ax2, ax3=ax3,
                        fig=fig2, x=x, y1=y1, y2=y2)

        # print(f"current Information Gain: {IG}, largest IG yet: {maxIG}")
        pointsInBudget = getPointsInBudget(UHP_gdf, SHP_gdf, sensorNames, energyBudget, joulesPerMeter, joulesPerSecond,
                                           dataSize, transferRate)

        if len(UHP_gdf) == 0:
            arProb = totalEnergy / energyBudget
        else:
            arProb = (totalEnergy / energyBudget) * (
                    1 - (len(pointsInBudget) / len(UHP_gdf)))  # as we approach budget, more likely to remove
        if len(UHP_gdf) + len(SHP_gdf) != len(HP_gdf):
            print('ERROR: SELECTED + UNSELECTED != HP')
            print(len(SHP_gdf), '+', len(UHP_gdf), ' != ', len(HP_gdf))
            exit(1)
    if not exhaustive:
        # print('Checking MSE of 10 hoverpoint sets that had the lowest entropy\n')
        # iterationOfBest = 0
        # tenLowestEntropys = sorted(entropyToSHP.keys())[:20]  # RIGHT NOW IS CHECKING 20 LOWEST ENTROPIES
        # minMSE = 999
        # for entropy in tenLowestEntropys:
        #     selected = entropyToSHP[entropy]
        #     features = getSensorNames(selected['geometry'], sensorNames)
        #     mse = getMSE(features, originalDF)
        #     print(f'mse: {mse}, previous minMSE: {minMSE}')
        #     if mse < minMSE:
        #         minMSE = mse
        #         bestSHP = selected.copy()
        features = getSensorNames(bestSHP.geometry, sensorNames)
        minMSE = getMSE(features, originalDF)
    else:
        features = getSensorNames(bestSHP.geometry, sensorNames)
        minMSE = getMSE(features, originalDF)
    features = getSensorNames(SHP_gdf.geometry, sensorNames)
    finalMSE2 = getMSE(features, originalDF)
    # get the cheapest path of best shp, because TSP is heuristic and gives different paths, some better than others
    minEnergy = 999999999999
    for i in range(10):
        tempBestSHP = bestSHP.copy()
        tempBestSHP, _ = findMinTravelDistance(tempBestSHP, scramble=True)
        tempBestSHP, distance = getEnergy(tempBestSHP, sensorNames, joulesPerMeter, joulesPerSecond, dataSize,
                                          transferRate)
        print(f"energy of best {i}: {tempBestSHP['energy'][0]}")
        if tempBestSHP['energy'][0] < minEnergy:
            minEnergy = tempBestSHP['energy'][0]
            bestSHP = tempBestSHP.copy()
            distanceOfBest = distance
    print(f"EPSILON-GREEDY ALGORITHM WITH {numLoops} LOOPS AND ENERGY WEIGHT {energyWeight}")
    printResults(bestSHP, iterationOfBest, distanceOfBest, minMSE, sensorNames)
    ax2.scatter(iterationOfBest, minMSE, color='red', zorder=10)
    plotPath(ax1, bestSHP)
    bestSHP.plot(ax=ax1, color='red', markersize=10, alpha=1)
    printTime(startTime)
    for key, value in sensorNames.items():
        if len(value) == 1:
            ax1.text(key.x, key.y, str(value[0]), fontsize=10, ha='center', va='bottom')
    if savePlots:
        writeResults(bestSHP, iterationOfBest, distanceOfBest, minMSE, sensorNames, outputTextName, startTime, finalMSE2)
        fig1.savefig(pathPlotName, bbox_inches='tight')
        fig2.savefig(msePlotName, bbox_inches='tight')
    else:
        plt.show()

if __name__ == '__main__':
    # training_data = pd.read_csv('training_data0514.csv', index_col=0)
    soil_seeds = [1, 3, 4, 5, 7, 8, 9, 10, 12, 13]
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
            epsilonGreedy(fig1=fig, ax1=ax, HP_gdf=hoverPoints, sensorNames=sensorNames, df=training_data, numLoops=200,
                        startTime=time.time(), savePlots=True, pathPlotName=f'Energy Budget Experiments Soils IG/seed{seed}/pathPlot{budget_str}.png', msePlotName=f'Energy Budget Experiments Soils IG/seed{seed}/msePlot{budget_str}.png',
                        outputTextName=f'Energy Budget Experiments Soils IG/seed{seed}/output{budget_str}.txt', energyWeight=0, energyBudget=budget, joulesPerMeter=10,
                        joulesPerSecond=35, dataSize=100, transferRate=9, exhaustive=False)
