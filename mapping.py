import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import pyproj
from shapely.geometry import Point
import TSP
import numpy as np
from SetCoverPy import setcover
from ML import processData, getSyntheticDF, getSyntheticDF_two, discretizeData, calculateEntropy,normalizeData
import random
from scipy.stats import multivariate_normal



# turn shp file, which is in long/lat degrees into meters xy axis. treating 0,0 as the depot
def processSHP(filePath, df, height, commRadius):
    sensors = gpd.read_file(filePath)
    sensors = simplifyPlot(sensors)
    # Rename the second and third columns
    new_column_names = ['x', 'y']
    sensors = sensors.rename(columns={sensors.columns[1]: new_column_names[0],
                                      sensors.columns[2]: new_column_names[1]})
    fig, ax = plt.subplots(figsize=(8, 8))

    # Zoom out plot
    padding = 100
    ax.set_xlim(min(sensors['x']) - padding, max(sensors['x']) + padding)
    ax.set_ylim(min(sensors['y']) - padding, max(sensors['y']) + padding)
    # Plot the sensor points
    sensors = sensors[sensors['Location'].isin(df.columns)]
    sensors.plot(ax=ax, color='blue', markersize=25, alpha=0.7)

    hoverPoints, sensorNames = getHoverPoints(sensors, commRadius, height, ax)
    # Set plot title and labels
    ax.set_title('Sensors with Communication Ranges (Meters)')
    ax.set_xlabel('East')
    ax.set_ylabel('North')
    return fig, ax, hoverPoints, sensorNames


# convert degrees to meters and shift numbers close to 0
def simplifyPlot(gdfInput):
    # Define the input and output coordinate systems
    input_crs = gdfInput.crs
    output_crs = pyproj.CRS.from_epsg(3857)  # EPSG code for WGS 84 Pseudo-Mercator

    # Perform the coordinate transformation from degrees to meters
    transformer = pyproj.Transformer.from_crs(input_crs, output_crs, always_xy=True)
    gdfInput['geometry'] = gdfInput['geometry'].to_crs(output_crs)
    gdfInput['Easting'], gdfInput['Northing'] = transformer.transform(gdfInput['Easting'].values,
                                                                      gdfInput['Northing'].values)

    # Normalize values
    subX = min(gdfInput['Easting']) - 50
    subY = min(gdfInput['Northing']) - 50
    gdfInput['Easting'] = gdfInput['Easting'] - subX
    gdfInput['Northing'] = gdfInput['Northing'] - subY
    # Define the lambda function to subtract values from a point
    subtract_from_point = lambda point: Point(point.x - subX, point.y - subY)
    # Apply the lambda function to the geometry column
    gdfInput['geometry'] = gdfInput['geometry'].apply(subtract_from_point)

    return gdfInput
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
    # indexToLocationSet = {}
    # for index, row in overlapsOf3.iterrows():
    #     locationValues = set(row[['Location_1_1', 'Location_2_1', 'Location_1_2', 'Location_2_2']])
    #     indexToLocationSet[index] = locationValues
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

    #This keeps multiple hover points at the same location, but causes problems due to the way getSensorNames is called
    # # remove duplicate hover-points by temporarily making a string column
    # hoverPoints['geometry_str'] = hoverPoints['geometry'].astype(str)
    # # hoverPoints = hoverPoints.drop_duplicates(subset='geometry_str')
    # pointsSet = set()
    # locationValues = set()
    # indexesToDrop = []
    # for index, row in hoverPoints.iterrows():
    #     if row['geometry_str'] in pointsSet and tuple(sorted(indexToLocationSet[index])) in locationValues:
    #         indexesToDrop.append(index)
    #     else:
    #         locationValues.add(tuple(sorted(indexToLocationSet[index])))
    #         pointsSet.add(row['geometry_str'])
    # hoverPoints = hoverPoints.drop(indexesToDrop)
    # hoverPoints = hoverPoints.drop('geometry_str', axis=1)
    # sensorNames = {}
    # for index, locations in indexToLocationSet.items():
    #     if index in hoverPoints.index:
    #         sensorNames[hoverPoints['geometry'][index]] = list(locations)
    # hoverPoints.reset_index(drop=True, inplace=True)
    # create dictionary to correspond hover points to sensors
    # sensorNames = {}
    # for hoverPoint in hoverPoints['geometry']:
    #     sensorNames[hoverPoint] = []
    #     # if hoverPoint in sensors['geometry']:
    #
    #     for circle in rangeCircles['geometry']:
    #         if hoverPoint.within(circle):
    #             sensorName = rangeCircles.loc[rangeCircles['geometry'] == circle, 'Location'].values[0]
    #             sensorNames[hoverPoint].append(sensorName)

    # groupedPoints = {}
    # for point, sensorList in sensorNames.items():
    #     sortedSensors = tuple(sorted(sensorList))
    #     if sortedSensors in groupedPoints:
    #         groupedPoints[sortedSensors].append(point)
    #     else:
    #         groupedPoints[sortedSensors] = [point]
    # filteredGroupedPoints = {sensors: points for sensors, points in groupedPoints.items() if len(points) > 1}
    # # Print the nested array containing points with the same sensor list
    # nested_array = list(filteredGroupedPoints.values())
    # for group in nested_array:
    #     print("Group:")
    #     for point in group:
    #         print(point)

    return hoverPoints, sensorNames


# height, commRadius,
def addSensorsUniformRandom(df, numSensors, areaLength=800, areaWidth=1300):
    # fig, ax = plt.subplots(figsize=(8, 8))
    # ax.set_xlim(-100, 1300 + 100)
    # ax.set_ylim(-100, 800 + 100)
    originalSeed = np.random.get_state()
    np.random.seed(42)
    sensors = gpd.read_file('CAF_Sensor_Dataset_2/CAF_sensors.shp')
    sensors = simplifyPlot(sensors)
    sensors = sensors[sensors['Location'].isin(df.columns)]
    minDistance = 70
    if numSensors < len(df.columns):
        columnNames = list(df.columns)
        random.shuffle(columnNames)
        columnsToKeep = columnNames[:numSensors]
        df = df[columnsToKeep]
        sensors = sensors[sensors['Location'].isin(df.columns)]
        sensorPoints = sensors['geometry']
    elif numSensors > len(df.columns):
        print('averaging closest points to generate extra sensors\n')
        sensorPoints = sensors['geometry']
        newSensorPoints = sensorPoints.tolist()
        numSensorsToAdd = numSensors - len(df.columns)
        newDF = df.copy()
        for i in range(numSensorsToAdd):
            while True:
                x = random.uniform(0, areaWidth)
                y = random.uniform(0, areaLength)
                newPoint = Point(x, y)
                # print(newPoint)
                if all(newPoint.distance(existingPoint) >= minDistance for existingPoint in newSensorPoints):
                    tempGDF = gpd.GeoDataFrame(geometry=sensorPoints)
                    tempGDF['distance'] = tempGDF['geometry'].apply(lambda point: newPoint.distance(point))
                    tempGDF = tempGDF.sort_values(by='distance')
                    closestPoints = tempGDF.iloc[:3]
                    sumOfDistances = sum(closestPoints['distance'])
                    weights = [sumOfDistances, sumOfDistances, sumOfDistances] / (closestPoints['distance'] ** 2) # PLAY AROUND WITH
                    # sensorRows = sensors[sensors['geometry'].isin(closestPoints['geometry'])]
                    normalizedWeights = weights / weights.sum()
                    closestSensors = sensors.loc[closestPoints.index]['Location']
                    # newDF[f'synthetic {i + 1}'] = (df[closestSensors].mul(weights_series, axis=1)).sum(axis=1)
                    newValues = [0] * len(newDF)
                    for sensor, normalizedWeight in zip(closestSensors, normalizedWeights):
                        newValues += df[sensor] * normalizedWeight
                    newDF[f'synthetic {i + 1}'] = newValues
                    newSensorPoints.append(newPoint)
                    break
        df = newDF
        sensorPoints = newSensorPoints
    sensorsGDF = gpd.GeoDataFrame(geometry=sensorPoints)
    sensorsGDF['Location'] = df.columns
    # sensorsGDF.plot(ax=ax, markersize=30, color='blue')
    # hoverPoints, sensorNames = getHoverPoints(sensorsGDF, commRadius, height, ax)
    np.random.set_state(originalSeed)
    # print(f'by the end of add... len(hover) = {len(hoverPoints)}, len(sensorNames) = {len(sensorNames)}')
    return sensorsGDF, df

def generateSensorsUniformRandom(height, commRadius, df, numSensors, areaLength=800, areaWidth=1300):
    minDistance = 70
    # df = getSyntheticDF(df)
    df = getSyntheticDF_two(df, numSensors=numSensors)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-100, areaWidth + 100)
    ax.set_ylim(-100, areaLength + 100)
    if numSensors < len(df.columns):
        columnsToKeep = df.columns[:numSensors]
        df = df[columnsToKeep]
    # place a sensor point and make its values the average of the three nearest points
    elif numSensors > len(df.columns):
        print('cant generate that many sensors')
        exit(1)

    else:
        sensorPoints = []
        for _ in range(numSensors):
            while True:
                x = random.uniform(0, areaWidth)
                y = random.uniform(0, areaLength)
                newPoint = Point(x, y)

                if all(newPoint.distance(existingPoint) >= minDistance for existingPoint in sensorPoints):
                    sensorPoints.append(Point(x, y))
                    break
    sensorsGDF = gpd.GeoDataFrame(geometry=sensorPoints)
    sensorsGDF['Location'] = df.columns
    sensorsGDF.plot(ax=ax, markersize=30, color='blue')
    hoverPoints, sensorNames = getHoverPoints(sensorsGDF, commRadius, height, ax)
    return fig, ax, hoverPoints, sensorNames, df


# input: gdf with hover points as geometry column
# output: gdf with ordered hover points and new 'distance' column to travel this path
def findMinTravelDistance(hoverPoints, scramble=False):
    points = np.array([(point.x, point.y) for point in hoverPoints.geometry])
    if not scramble:
        orderedHoverPoints, distance = TSP.hoverPointsTSP(points)
    else:
        orderedHoverPoints, distance = TSP.hoverPointsTSP(points, scramble=True)
    return orderedHoverPoints, distance


# plot lines between selected geometry points in the order provided
def plotPath(ax, hoverPoints):
    coordinates = np.array([(point.x, point.y) for point in hoverPoints.geometry])
    ax.plot([coordinates[0][0], 0], [coordinates[0][1], 0], 'r-')
    for i in range(len(coordinates) - 1):
        xValues = [coordinates[i][0], coordinates[i + 1][0]]
        yValues = [coordinates[i][1], coordinates[i + 1][1]]
        ax.plot(xValues, yValues, 'r-')
    ax.plot([coordinates[len(coordinates) - 1][0], 0], [coordinates[len(coordinates) - 1][1], 0], 'r-')  # return path


# Only keeps the smallest amount of hoverpoints needed to cover all the sensors, used in RSEO
def minSetCover(sensorNames, hoverPoints):
    original_random_state = random.getstate()
    orig_np_state = np.random.get_state()
    random.seed()
    np.random.seed()
    allSensors = sorted(set(sensor for sensors in sensorNames.values() for sensor in sensors))
    # Create a mapping of sensors to their indices in the matrix
    sensorIndexes = {sensor: i for i, sensor in enumerate(allSensors)}
    numRows = len(allSensors)
    numCols = len(sensorNames)
    if numCols != len(hoverPoints):
        hoverPoints['geometry_str'] = hoverPoints['geometry'].astype(str)
        hoverPoints = hoverPoints.drop_duplicates(subset='geometry_str')
        hoverPoints = hoverPoints.drop(columns='geometry_str')
    print(f'length of sensorNames: {numCols}, length of hoverPoints: {len(hoverPoints)}')
    aMatrix = np.zeros((numRows, numCols), dtype=int)
    for j, (point, sensors) in enumerate(sensorNames.items()):
        for sensor in sensors:
            i = sensorIndexes[sensor]
            aMatrix[i, j] = True
    sums = np.sum(aMatrix, axis=0)
    cost = [value - ((value - 1) * .1) for value in sums]
    solver = setcover.SetCover(amatrix=aMatrix, cost=cost)
    result = solver.SolveSCP()
    decisionList = solver.s
    filteredHoverPoints = hoverPoints[decisionList]
    filteredHoverPoints.reset_index(drop=True, inplace=True)
    validPoints = set(filteredHoverPoints['geometry'])
    filteredSensorNames = {point: value for point, value in sensorNames.items() if point in validPoints}
    random.setstate(original_random_state)
    np.random.set_state(orig_np_state)
    return filteredSensorNames, filteredHoverPoints


def getSensorNames(points, sensorNames):
    corrSensorsSet = set()
    for point in points:
        if point in sensorNames:
            corrSensorsSet.update(sensorNames[point])
    return list(corrSensorsSet)


def testMapping():
    # random.seed(42)
    # np.random.seed(42)
    pd.set_option('display.max_rows', 5)
    pd.set_option('display.max_columns', 100)
    shpFilePath = 'CAF_Sensor_Dataset_2/CAF_sensors.shp'
    communicationRadius = 70
    height = 15
    dataFolder = 'CAF_Sensor_Dataset_2/caf_sensors/Hourly'
    df = processData(dataFolder)
    np.random.seed(1)
    random.seed(1)
    fig, ax, sensorsGDF, df = addSensorsUniformRandom(height, communicationRadius, df, numSensors=80)
    df = df[np.random.permutation(df.columns)]
    df = df.iloc[:, :20]
    sensorsGDF = sensorsGDF[sensorsGDF['Location'].isin(df.columns)]
    hoverPoints, sensorNames = getHoverPoints(sensorsGDF, commRadius=70, ax=ax, height=15)
    hoverPoints.plot(ax=ax, color='red', markersize=25)
    plt.show()
    print(df)
    synthDF = normalizeData(df)
    print(synthDF)
    exit()
    # fig, ax, hoverPoints, sensorNames, df = generateSensorsUniformRandom(height, communicationRadius, df,
    #                                                                      numSensors=50)
    features = df.columns
    features = features[:30]
    # mi = getMutualInformation(features, df)
    # fig, ax, hoverPoints, sensorNames, df = addSensorsUniformRandom(height, communicationRadius, df,
    #                                                                      numSensors=50)

    # df = getSyntheticDF_two(df, numSensors=60)
    # fig, ax, hoverPoints, sensorNames, df = generateSensorsUniformRandom(height, communicationRadius, df,
    #                                                                      numSensors=50)

    # fig, ax, hoverPoints, sensorNames = processSHP(shpFilePath, df, height, communicationRadius)

    hoverPoints.plot(ax=ax, color='yellow', markersize=25)
    print(f'there are {len(hoverPoints)} hover points')
    print(df)
    for key, value in sensorNames.items():
        ax.text(key.x, key.y, str(len(value)), fontsize=10, ha='center', va='bottom')
    duplicates = set(set())
    for names in sensorNames.values():
        if set(names) in duplicates:
            print('theres a duplicate in sensor names')
            exit()
        else:
            duplicates.update(set(names))
    print('no dup sensors')

    # plt.show()
    exit()




    rSeed = 3
    random.seed(rSeed)
    np.random.seed(rSeed)
    # pathPlotName = f"Experiment Maps/Sensors37Seed{rSeed}"
    fig, ax, hoverPoints, sensorNames, df = generateSensorsUniformRandom(height, communicationRadius, df,
                                                        numSensors=50)
    print(df)
    hoverPoints.plot(ax=ax, color='yellow', markersize=25)
    count = 0

    print(f'there are {len(hoverPoints)} hover points')
    plt.show()


# testMapping()