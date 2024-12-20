from mapping import processSHP, findMinTravelDistance, generateSensorsUniformRandom, getSensorNames, minSetCover, addSensorsUniformRandom
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from ML import getMSE, processData, getConditionalEntropy, calculateEntropy, getInformationGain
import random
from joblib import Parallel, delayed
import time
import numpy as np


def processFiles(height, communicationRadius, minSet, generate, addToOriginal, numSensors):
    positionFilePath = 'CAF_Sensor_Dataset_2/CAF_sensors.shp'
    dataFolderPath = 'CAF_Sensor_Dataset_2/caf_sensors/Hourly'
    df = processData(dataFolderPath)
    if generate:
        if addToOriginal:
            fig, ax, hoverPoints, sensorNames, df = addSensorsUniformRandom(height, communicationRadius, df,
                                                                            numSensors)
        else:
            fig, ax, hoverPoints, sensorNames, df = generateSensorsUniformRandom(height, communicationRadius, df, numSensors)
    else:
        fig, ax, hoverPoints, sensorNames = processSHP(positionFilePath, df, height, communicationRadius)

    if minSet:
        print(f'hover points length: {len(hoverPoints)}')
        for i in range(5):
            sensorNames, hoverPoints = minSetCover(sensorNames, hoverPoints)
        print(f'hover points length: {len(hoverPoints)}')
    # hoverPoints.plot(ax=ax, color='yellow', markersize=10, alpha=1)
    # hoverPoints = hoverPoints.sample(n=20)  # Use this to test, reduces runtime
    sensorsWithData = set(df.columns)
    filteredDictionary = {}
    for point, sensors in sensorNames.items():
        filteredSensors = [sensor for sensor in sensors if sensor in sensorsWithData]
        if filteredSensors:
            filteredDictionary[point] = filteredSensors
    sensorNames = filteredDictionary
    hoverPoints = hoverPoints.loc[hoverPoints.geometry.isin(sensorNames.keys())]
    if len(hoverPoints) != len(sensorNames.values()):
        print(f"points in hpgdf = {len(hoverPoints.columns)}, sensorNames values = {len(sensorNames.values())}")
        print('ERROR')
    selected = gpd.GeoDataFrame()
    selected['geometry'] = None
    selected = selected.set_geometry('geometry')
    unselected = hoverPoints.copy()
    selected.crs = 'EPSG:3857'  # pseudo-mercator
    unselected.crs = 'EPSG:3857'
    return fig, ax, hoverPoints, unselected, selected, sensorNames, df


# get the total energy cost to travel to all the selected hover points and collect data from them
def getEnergy(selected, sensorNames, joulesPerMeter, joulesPerSecond, dataSize, transferRate, parallel=True):
    energy = 0
    selected, distance = findMinTravelDistance(selected)
    energy += distance * joulesPerMeter
    for hoverPoint in selected['geometry']:
        numCorrespondingSensors = len(sensorNames[hoverPoint])
        if parallel:
            timeToTransfer = dataSize / transferRate
        else:
            timeToTransfer = numCorrespondingSensors * dataSize / transferRate
        energy += timeToTransfer * joulesPerSecond
    selected['energy'] = energy
    return selected, distance


# determine what points can be visited without violating Energy budget
def getPointsInBudget(unselected, selected, sensorNames, energyBudget, joulesPerMeter, joulesPerSecond, dataSize,
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
    return inBudgetHP


# Adds a random hoverpoint that is in budget (IB)
def addRandomHP(unselected, unselectedIB, selected):
    print('ADD RANDOM')
    if unselectedIB.empty:
        print('No more hover points within budget')
        return unselected, selected
    randomRow = unselectedIB.sample(n=1)
    randomRow.rename(columns={'energyIfAdded': 'energy'}, inplace=True)
    randomRow = randomRow.reset_index(drop=True)
    pointAdded = randomRow.iloc[0]['geometry']
    unselected = unselected[unselected['geometry'] != pointAdded].reset_index(drop=True)
    SHP = gpd.GeoDataFrame(pd.concat([selected, randomRow], ignore_index=True), crs=selected.crs)
    SHP['energy'] = randomRow['energy'][0]
    return unselected, SHP


# Remove a random hover point from selected
def remRandomHP(unselected, selected):
    print('REMOVE RANDOM')
    if selected.empty:
        print('selected is empty')
        return selected
    rowToMove = selected.sample(n=1)
    del rowToMove['energy']
    UHP = gpd.GeoDataFrame(pd.concat([unselected, rowToMove], ignore_index=True), crs=unselected.crs)
    # if 'energy' in UHP.columns:
    #     UHP.drop('energy', axis=1, inplace=True)
    selected = selected.drop(index=rowToMove.index)
    return UHP, selected


# Add the best in-budget-hoverpoint to selected, best = max((oldEnergy - newEnergy) * (oldMSE - newMSE))

def addBestHP(unselected, unselectedIB, selected, sensorNames, df, energyWeight, exhaustive):
    print("ADD BEST")
    rewards = [-10] * len(unselectedIB)
    if unselectedIB.empty:
        print('No more hover points within budget')
        return unselected, selected
    if selected.empty:  # never happens in actual greedy alg
        oldEnergy = 0
        if exhaustive:
            oldMSE = 999999999
        else:
            oldIG = float('-inf')
    else:
        oldEnergy = selected['energy'][0]
        oldFeatures = getSensorNames(selected['geometry'], sensorNames)
        if exhaustive:
            oldMSE = getMSE(oldFeatures, df)
        else:
            oldIG = getInformationGain(oldFeatures, df)


    def calculateIGAdding(index, row):
        extractedRow = gpd.GeoDataFrame([row], geometry='geometry', crs=unselectedIB.crs)
        tempSHP = gpd.GeoDataFrame(pd.concat([selected, extractedRow], ignore_index=True), crs=unselectedIB.crs)
        features = getSensorNames(tempSHP['geometry'], sensorNames)
        informationGain = getInformationGain(features, df)  # for minimizing entropy
        newEnergy = unselectedIB['energyIfAdded'][index]
        return index, informationGain, newEnergy


    def calculateMSEAdding(index, row):
        # print('index ', index + 1, 'of', len(unselectedIB))
        extractedRow = gpd.GeoDataFrame([row], geometry='geometry', crs=unselectedIB.crs)
        tempSHP = gpd.GeoDataFrame(pd.concat([selected, extractedRow], ignore_index=True), crs=unselectedIB.crs)
        features = getSensorNames(tempSHP['geometry'], sensorNames)
        newMSE = getMSE(features, df)
        newEnergy = unselectedIB['energyIfAdded'][index]
        return index, newMSE, newEnergy

    if exhaustive:
        indexMSEEnergy = Parallel(n_jobs=-1)(delayed(calculateMSEAdding)
                                         (index, row) for index, row in unselectedIB.iterrows())
        indexMSEEnergy = sorted(indexMSEEnergy, key=lambda x: x[0])
    else:
        indexIGEnergy = Parallel(n_jobs=-1)(delayed(calculateIGAdding)
                                         (index, row) for index, row in unselectedIB.iterrows())
        indexIGEnergy = sorted(indexIGEnergy, key=lambda x: x[0])

    if exhaustive:
        newIndexMSEEnergy = []
        # calculate change in MSE and Energy
        for index, newMSE, newEnergy in indexMSEEnergy:
            changeMSE = oldMSE - newMSE
            changeEnergy = oldEnergy - newEnergy
            newTuple = (index, changeMSE, changeEnergy)
            newIndexMSEEnergy.append(newTuple)

        minMSE = min(value[1] for value in newIndexMSEEnergy)
        maxMSE = max(value[1] for value in newIndexMSEEnergy)
        minEnergy = min(value[2] for value in newIndexMSEEnergy)
        maxEnergy = max(value[2] for value in newIndexMSEEnergy)
        for index, newMSE, newEnergy in newIndexMSEEnergy:
            normalizedNewMSE = (newMSE - minMSE) / (maxMSE - minMSE) if maxMSE != minMSE else 0
            normalizedNewEnergy = (newEnergy - minEnergy) / (maxEnergy - minEnergy) if maxEnergy != minEnergy else 0
            reward = normalizedNewEnergy * energyWeight + normalizedNewMSE
            rewards[index] = reward
    else:
        for index, IG, newEnergy in indexIGEnergy:
            rewards[index] = IG

    if exhaustive:
        # make sure not to add a redundant point
        for index, newMSE, newEnergy in indexMSEEnergy:
            if newMSE == oldMSE:
                rewards[index] = -999
        maxReward = max(rewards)
        indexesEnergiesOfBest = []
        # get all the points resulting in the same mse
        for index, newMSE, newEnergy in indexMSEEnergy:
            if rewards[index] == maxReward:
                indexesEnergiesOfBest.append((index, newEnergy))
    else:
        # make sure not to add a redundant point
        for index, IG, newEnergy in indexIGEnergy:
            if IG == oldIG:
                rewards[index] = -999
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


def addBestHybrid(unselected, unselectedIB, selected, sensorNames, originalDF, discreteDF, oldMSE):
    print("ADD BEST")
    rewards = [-999] * len(unselectedIB)
    if unselectedIB.empty:
        print('No more hover points within budget')
        return unselected, selected, oldMSE
    if selected.empty:  # never happens in actual greedy alg
        oldEntropy = 999
        oldEnergy = 0
    else:
        oldEnergy = selected['energy'][0]
        oldFeatures = getSensorNames(selected['geometry'], sensorNames)
        oldEntropy = getConditionalEntropy(oldFeatures, discreteDF)

    def calculateIGAdding(index, row):
        extractedRow = gpd.GeoDataFrame([row], geometry='geometry', crs=unselectedIB.crs)
        tempSHP = gpd.GeoDataFrame(pd.concat([selected, extractedRow], ignore_index=True), crs=unselectedIB.crs)
        features = getSensorNames(tempSHP['geometry'], sensorNames)
        newEntropy = getConditionalEntropy(features, discreteDF)
        informationGain = oldEntropy - newEntropy  # for minimizing entropy
        return index, informationGain

    if len(unselectedIB) > 10:
        # Calculate the information gain for every unselected hovering point
        indexIG = Parallel(n_jobs=-1)(delayed(calculateIGAdding)
                                            (index, row) for index, row in unselectedIB.iterrows())
        indexIG = sorted(indexIG, key=lambda x: x[1], reverse=True)

        # check the mse of the ten hovering points that resulted in the lowest mse
        tenHighestIGindices = [item[0] for item in indexIG[:10]]
        filteredUnselectedIB = unselectedIB.loc[tenHighestIGindices]
    else:
        filteredUnselectedIB = unselectedIB.copy()

    def calculateMSEAdding(index, row):
        # print('index ', index + 1, 'of', len(unselectedIB))
        extractedRow = gpd.GeoDataFrame([row], geometry='geometry', crs=unselectedIB.crs)
        tempSHP = gpd.GeoDataFrame(pd.concat([selected, extractedRow], ignore_index=True), crs=unselectedIB.crs)
        features = getSensorNames(tempSHP['geometry'], sensorNames)
        newMSE = getMSE(features, originalDF)
        newEnergy = unselectedIB['energyIfAdded'][index]
        return index, newMSE, newEnergy

    indexMSEEnergy = Parallel(n_jobs=-1)(delayed(calculateMSEAdding)
                                         (index, row) for index, row in filteredUnselectedIB.iterrows())
    indexMSEEnergy = sorted(indexMSEEnergy, key=lambda x: x[0])

    newIndexMSEEnergy = []
    # calculate change in MSE and Energy
    for index, newMSE, newEnergy in indexMSEEnergy:
        changeMSE = oldMSE - newMSE
        changeEnergy = oldEnergy - newEnergy
        newTuple = (index, changeMSE, changeEnergy)
        newIndexMSEEnergy.append(newTuple)
        if changeMSE == 0:
            rewards[index] = -999
        else:
            rewards[index] = changeMSE

    maxReward = max(rewards)
    indexesEnergiesOfBest = []
    # get all the points resulting in the same mse
    for index, newMSE, newEnergy in indexMSEEnergy:
        if rewards[index] == maxReward:
            bestMSE = newMSE
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
                return updatedUnselected, updatedSelected, bestMSE
    for index, mse, energy in indexMSEEnergy:
        if index == indexOfBest:
            newMSE = mse
    print('Normal add')
    rowToMove = unselectedIB.loc[[indexOfBest]].copy()
    rowToMove.rename(columns={'energyIfAdded': 'energy'}, inplace=True)
    rowToMove = rowToMove.reset_index(drop=True)
    selected['energy'] = rowToMove['energy'][0]
    rowToMovePoint = rowToMove['geometry'].iloc[0]
    unselected = unselected[unselected['geometry'] != rowToMovePoint]
    selected = gpd.GeoDataFrame(pd.concat([selected, rowToMove], ignore_index=True), crs=unselectedIB.crs)
    return unselected, selected, newMSE


# Remove the best hover-point from selected
def remBestHP(unselected, selected, sensorNames, df, energyWeight, joulesPerMeter,
              joulesPerSecond, dataSize, transferRate, exhaustive):
    print('REMOVE BEST')
    rewards = [-999] * len(selected)
    if selected.empty:  # shouldn't happen in actual greedy alg, used for testing
        return unselected, selected
    elif len(selected) == 1:
        return remRandomHP(unselected, selected)
    else:
        oldEnergy = selected['energy'][0]
        oldFeatures = getSensorNames(selected['geometry'], sensorNames)
        if exhaustive:
            oldMSE = getMSE(oldFeatures, df)
        else:
            oldIG = getInformationGain(oldFeatures, df)
    # Check for any redundant hoverPoints
    selectedSensors = []
    for point in selected['geometry']:
        selectedSensors.extend(sensorNames[point])
    sensorsWithDuplicates = []
    for sensor in selectedSensors:
        if selectedSensors.count(sensor) > 1 and sensor not in sensorsWithDuplicates:
            sensorsWithDuplicates.append(sensor)
    if len(sensorsWithDuplicates) > 0:
        redundancies = True
    else:
        redundancies = False
    def calculateIGRemoving(index):
        # print('index ', index + 1, 'of', len(selected))
        tempSHP = selected.drop(index=index).reset_index(drop=True)
        tempSHP, _ = getEnergy(tempSHP, sensorNames, joulesPerMeter, joulesPerSecond, dataSize, transferRate)
        newEnergy = tempSHP['energy'][0]
        features = getSensorNames(tempSHP['geometry'], sensorNames)
        informationGain = getInformationGain(features, df)  # for minimizing entropy
        return index, informationGain, newEnergy


    def calculateMSERemoving(index):
        # print('index ', index + 1, 'of', len(selected))
        tempSHP = selected.drop(index=index).reset_index(drop=True)
        tempSHP, _ = getEnergy(tempSHP, sensorNames, joulesPerMeter, joulesPerSecond, dataSize, transferRate)
        features = getSensorNames(tempSHP['geometry'], sensorNames)
        newMSE = getMSE(features, df)
        newEnergy = tempSHP['energy'][0]
        return index, newMSE, newEnergy
    if exhaustive:
        indexMSEEnergy = Parallel(n_jobs=-1)(delayed(calculateMSERemoving)
                                             (index) for index, _ in selected.iterrows())
        indexMSEEnergy = sorted(indexMSEEnergy, key=lambda x: x[0])
    else:
        indexIGEnergy = Parallel(n_jobs=-1)(delayed(calculateIGRemoving)
                                             (index) for index, _ in selected.iterrows())
        indexIGEnergy = sorted(indexIGEnergy, key=lambda x: x[0])
    if exhaustive:
        newIndexMSEEnergy = []
        # calculate change in MSE and Energy
        for index, newMSE, newEnergy in indexMSEEnergy:
            changeMSE = oldMSE - newMSE
            changeEnergy = oldEnergy - newEnergy
            newTuple = (index, changeMSE, changeEnergy)
            newIndexMSEEnergy.append(newTuple)

        minMSE = min(value[1] for value in newIndexMSEEnergy)
        maxMSE = max(value[1] for value in newIndexMSEEnergy)
        minEnergy = min(value[2] for value in newIndexMSEEnergy)
        maxEnergy = max(value[2] for value in newIndexMSEEnergy)
        for index, newMSE, newEnergy in newIndexMSEEnergy:
            normalizedNewMSE = (newMSE - minMSE) / (maxMSE - minMSE) if maxMSE != minMSE else 0
            normalizedNewEnergy = (newEnergy - minEnergy) / (maxEnergy - minEnergy) if maxEnergy != minEnergy else 0
            reward = normalizedNewEnergy * energyWeight + normalizedNewMSE
            rewards[index] = reward
    else:
        for index, IG, newEnergy in indexIGEnergy:
            rewards[index] = IG

    # When there are redundant features, only keep one that results in largest MSE
    # first trials use if redundancies and not good removes
    if redundancies:
        print('REDUNDANCY CHECK')
        # create nested list of indexes of points in range of a sensor collected 2+ times, and make sure no duplicates
        # i.e. [[1,2], [2,3]] isnt allowed
        indexesOfSensors = [[] for _ in range(len(sensorsWithDuplicates))]
        for i, duplicateSensor in enumerate(sensorsWithDuplicates):
            for idx, row in selected.iterrows():
                if duplicateSensor in sensorNames[row['geometry']]:
                    indexesOfSensors[i].append(idx)
        filteredIndexes = []
        seenIndexes = set()
        for sublist in indexesOfSensors:
            if not any(index in seenIndexes for index in sublist):
                seenIndexes.update(sublist)
                filteredIndexes.append(sublist)
        indexesOfSensors = filteredIndexes

        indexesToDrop = set()
        indexesToAdd = None
        for indexList in indexesOfSensors:
            energyList = []
            if exhaustive:
                for idx in indexList:
                    for index, newMSE, newEnergy in indexMSEEnergy:
                        if idx == index:
                            energyList.append(newEnergy)
            else:
                for idx in indexList:
                    for index, newMI, newEnergy in indexIGEnergy:
                        if idx == index:
                            energyList.append(newEnergy)
            tempSHP = selected.copy()
            tempUHP = unselected.copy()
            rowsToCheck = tempSHP.loc[list(indexList)].copy()
            assocSensors = getSensorNames(rowsToCheck['geometry'], sensorNames)
            rowsToCheck.drop(columns=['energy'], inplace=True)
            tempUHP = gpd.GeoDataFrame(pd.concat([tempUHP, rowsToCheck], ignore_index=True), crs=unselected.crs)
            rowsToCheck['energyIfAdded'] = energyList
            tempSHP = tempSHP.drop(index=list(indexList)).reset_index(drop=True)
            # check for a point that covers all the redundant points, if there is one and it isn't already in
            # rowsToCheck, add it
            pointToAdd = None
            for key, namesList in sensorNames.items():
                sortedNamesList = sorted(namesList)
                sortedAssocSensors = sorted(assocSensors)
                if sortedNamesList == sortedAssocSensors:
                    if not any(geom.equals(key) for geom in rowsToCheck['geometry']):
                        pointToAdd = key
                        pointToAddData = {'geometry': [pointToAdd], 'energy': oldEnergy}
                        pointToAddRow = gpd.GeoDataFrame(pointToAddData, crs=tempSHP.crs)
                        tempTempSHP = gpd.GeoDataFrame(pd.concat([tempSHP, pointToAddRow], ignore_index=True),
                                                       crs=tempSHP.crs)
                        tempTempSHP, _ = getEnergy(tempTempSHP, sensorNames, joulesPerMeter, joulesPerSecond, dataSize,
                                                   transferRate)
                        newEnergyIfAdded = tempTempSHP['energy'][0]
                        pointToAddData = {'geometry': [pointToAdd], 'energyIfAdded': [newEnergyIfAdded]}
                        pointToAddRow = gpd.GeoDataFrame(pointToAddData, crs=tempSHP.crs)
                        rowsToCheck = gpd.GeoDataFrame(pd.concat([rowsToCheck, pointToAddRow], ignore_index=True),
                                                       crs=rowsToCheck.crs)

            rowsToCheck = rowsToCheck.reset_index(drop=True)
            tempUHP, tempSHP = addBestHP(unselected=tempUHP, unselectedIB=rowsToCheck, selected=tempSHP,
                                         energyWeight=energyWeight, df=df, sensorNames=sensorNames, exhaustive=exhaustive)
            # if we added a new overlap point
            if pointToAdd is not None:
                indexesToAdd = set()
                indexesToAdd.update([index for index, geom in enumerate(unselected['geometry']) if geom == pointToAdd])
                indexesToDrop.update(indexList)

            else:
                droppedIndexes = selected[~selected['geometry'].isin(tempSHP['geometry'])].index
                indexesToDrop.update(droppedIndexes)
        rowsToDrop = selected.loc[list(indexesToDrop)].copy()
        rowsToDrop.drop(columns=['energy'])
        if indexesToAdd is not None:
            rowsToAdd = unselected.loc[list(indexesToAdd)].copy()
            unselected = unselected.drop(indexesToAdd)
            selected = selected.drop(indexesToDrop)
            unselected = gpd.GeoDataFrame(pd.concat([unselected, rowsToDrop], ignore_index=True), crs=unselected.crs)
            unselected.reset_index(drop=True, inplace=True)
            selected = gpd.GeoDataFrame(pd.concat([selected, rowsToAdd], ignore_index=True), crs=unselected.crs)
            selected.reset_index(drop=True, inplace=True)
            return unselected, selected
        else:
            unselected = gpd.GeoDataFrame(pd.concat([unselected, rowsToDrop], ignore_index=True), crs=unselected.crs)
            selected.reset_index(drop=True, inplace=True)
            selected = selected.drop(indexesToDrop).reset_index(drop=True)
            return unselected, selected

    maxReward = max(rewards)
    indexesEnergiesOfBest = []
    if exhaustive:
        for index, newMSE, newEnergy in indexMSEEnergy:
            if rewards[index] == maxReward:
                indexesEnergiesOfBest.append((index, newEnergy))
    else:
        for index, IG, newEnergy in indexIGEnergy:
            if rewards[index] == maxReward:
                indexesEnergiesOfBest.append((index, newEnergy))
    if len(indexesEnergiesOfBest) > 0:
        indexOfBest, energyOfBest = indexesEnergiesOfBest[0]
    else:
        print('Something went wrong...')
        print(f"rewards\n {rewards}")
        print(f"indexMSEEnergy\n {indexMSEEnergy}")
        exit(0)


    selected['energy'] = energyOfBest
    rowToMove = selected.loc[[indexOfBest]].copy()
    unselected = gpd.GeoDataFrame(pd.concat([unselected, rowToMove], ignore_index=True), crs=unselected.crs)
    selected = selected.drop(indexOfBest).reset_index(drop=True)
    return unselected, selected


def remBestHybrid(unselected, selected, sensorNames, originalDF, discreteDF, joulesPerMeter,
                  joulesPerSecond, dataSize, transferRate, oldMSE):
    print('REMOVE BEST')
    rewards = [-999] * len(selected)
    if selected.empty:  # shouldn't happen in actual greedy alg, used for testing
        return unselected, selected
    elif len(selected) == 1:
        return remRandomHP(unselected, selected)
    else:
        oldEnergy = selected['energy'][0]
        features = getSensorNames(selected['geometry'], sensorNames)
        oldEntropy = getConditionalEntropy(features, discreteDF)
    # Check for any redundant hoverPoints
    selectedSensors = []
    for point in selected['geometry']:
        selectedSensors.extend(sensorNames[point])
    sensorsWithDuplicates = []
    for sensor in selectedSensors:
        if selectedSensors.count(sensor) > 1 and sensor not in sensorsWithDuplicates:
            sensorsWithDuplicates.append(sensor)
    redundancies = len(sensorsWithDuplicates) > 0
    if redundancies:
        # When there are redundant features, only keep one that results in largest MSE
        print('REDUNDANCY CHECK')
        # create nested list of indexes of points in range of a sensor collected 2+ times, and make sure no duplicates
        # i.e. [[1,2], [2,3]] isnt allowed
        indexesOfSensors = [[] for _ in range(len(sensorsWithDuplicates))]
        for i, duplicateSensor in enumerate(sensorsWithDuplicates):
            for idx, row in selected.iterrows():
                if duplicateSensor in sensorNames[row['geometry']]:
                    indexesOfSensors[i].append(idx)
        filteredIndexes = []
        seenIndexes = set()
        for sublist in indexesOfSensors:
            if not any(index in seenIndexes for index in sublist):
                seenIndexes.update(sublist)
                filteredIndexes.append(sublist)
        indexesOfSensors = filteredIndexes

        indexesToDrop = set()
        indexesToAdd = None
        # I CAN IMPROVE THIS BY NOT RELYING ON INDEXMSEENERGY
        print(f'indexes of sensors: {indexesOfSensors}')

        for indexList in indexesOfSensors:
            energyList = []
            for idx in indexList:
                tempSHP = selected.drop(index=idx).reset_index(drop=True)
                tempSHP, _ = getEnergy(tempSHP, sensorNames, joulesPerMeter, joulesPerSecond, dataSize,
                                       transferRate)
                energyList.append(tempSHP['energy'][0])
            print(f'energy list: {energyList}')
            tempSHP = selected.copy()
            tempUHP = unselected.copy()
            rowsToCheck = tempSHP.loc[list(indexList)].copy()
            assocSensors = getSensorNames(rowsToCheck['geometry'], sensorNames)
            rowsToCheck.drop(columns=['energy'], inplace=True)
            tempUHP = gpd.GeoDataFrame(pd.concat([tempUHP, rowsToCheck], ignore_index=True), crs=unselected.crs)
            rowsToCheck['energyIfAdded'] = energyList
            tempSHP = tempSHP.drop(index=list(indexList)).reset_index(drop=True)
            # check for a point that covers all the redundant points, if there is one and it isn't already in
            # rowsToCheck, add it
            pointToAdd = None
            for key, namesList in sensorNames.items():
                sortedNamesList = sorted(namesList)
                sortedAssocSensors = sorted(assocSensors)
                if sortedNamesList == sortedAssocSensors:
                    if not any(geom.equals(key) for geom in rowsToCheck['geometry']):
                        pointToAdd = key
                        pointToAddData = {'geometry': [pointToAdd], 'energy': oldEnergy}
                        pointToAddRow = gpd.GeoDataFrame(pointToAddData, crs=tempSHP.crs)
                        tempTempSHP = gpd.GeoDataFrame(pd.concat([tempSHP, pointToAddRow], ignore_index=True),
                                                       crs=tempSHP.crs)
                        tempTempSHP, _ = getEnergy(tempTempSHP, sensorNames, joulesPerMeter, joulesPerSecond,
                                                   dataSize,
                                                   transferRate)
                        newEnergyIfAdded = tempTempSHP['energy'][0]
                        pointToAddData = {'geometry': [pointToAdd], 'energyIfAdded': [newEnergyIfAdded]}
                        pointToAddRow = gpd.GeoDataFrame(pointToAddData, crs=tempSHP.crs)
                        rowsToCheck = gpd.GeoDataFrame(pd.concat([rowsToCheck, pointToAddRow], ignore_index=True),
                                                       crs=rowsToCheck.crs)

            rowsToCheck = rowsToCheck.reset_index(drop=True)
            tempUHP, tempSHP, newMSE = addBestHybrid(unselected=tempUHP, unselectedIB=rowsToCheck, selected=tempSHP,
                                                     originalDF=originalDF, discreteDF=discreteDF,
                                                     sensorNames=sensorNames,
                                                     oldMSE=oldMSE)
            # if we added a new overlap point
            if pointToAdd is not None:
                indexesToAdd = set()
                indexesToAdd.update(
                    [index for index, geom in enumerate(unselected['geometry']) if geom == pointToAdd])
                indexesToDrop.update(indexList)

            else:
                droppedIndexes = selected[~selected['geometry'].isin(tempSHP['geometry'])].index
                indexesToDrop.update(droppedIndexes)
        rowsToDrop = selected.loc[list(indexesToDrop)].copy()
        rowsToDrop.drop(columns=['energy'])
        if indexesToAdd is not None:
            rowsToAdd = unselected.loc[list(indexesToAdd)].copy()
            unselected = unselected.drop(indexesToAdd)
            selected = selected.drop(indexesToDrop)
            unselected = gpd.GeoDataFrame(pd.concat([unselected, rowsToDrop], ignore_index=True),
                                          crs=unselected.crs)
            unselected.reset_index(drop=True, inplace=True)
            selected = gpd.GeoDataFrame(pd.concat([selected, rowsToAdd], ignore_index=True), crs=unselected.crs)
            selected.reset_index(drop=True, inplace=True)
            return unselected, selected, newMSE
        else:
            unselected = gpd.GeoDataFrame(pd.concat([unselected, rowsToDrop], ignore_index=True),
                                          crs=unselected.crs)
            selected.reset_index(drop=True, inplace=True)
            selected = selected.drop(indexesToDrop).reset_index(drop=True)
            return unselected, selected, newMSE
        redundancies = True
    else:

        def calculateIGRemoving(index):
            # print('index ', index + 1, 'of', len(selected))
            tempSHP = selected.drop(index=index).reset_index(drop=True)
            features = getSensorNames(tempSHP['geometry'], sensorNames)
            newEntropy = getConditionalEntropy(features, discreteDF)
            informationGain = oldEntropy - newEntropy  # for minimizing entropy
            return index, informationGain

        if len(selected) > 10:
            indexIG = Parallel(n_jobs=-1)(delayed(calculateIGRemoving)
                                                (index) for index, _ in selected.iterrows())
            indexIGEnergy = sorted(indexIG, key=lambda x: x[1], reverse=True)
            tenHighestIGindices = [item[0] for item in indexIGEnergy[:10]]
            filteredSelected = selected.loc[tenHighestIGindices]
        else:
            filteredSelected = selected.copy()

        def calculateMSERemoving(index):
            # print('index ', index + 1, 'of', len(selected))
            tempSHP = selected.drop(index=index).reset_index(drop=True)
            tempSHP, _ = getEnergy(tempSHP, sensorNames, joulesPerMeter, joulesPerSecond, dataSize, transferRate)
            features = getSensorNames(tempSHP['geometry'], sensorNames)
            newMSE = getMSE(features, originalDF)
            newEnergy = tempSHP['energy'][0]
            return index, newMSE, newEnergy

        indexMSEEnergy = Parallel(n_jobs=-1)(delayed(calculateMSERemoving)
                                                 (index) for index, _ in filteredSelected.iterrows())
        indexMSEEnergy = sorted(indexMSEEnergy, key=lambda x: x[0])

        newIndexMSEEnergy = []
        # calculate change in MSE and Energy
        for index, newMSE, newEnergy in indexMSEEnergy:
            changeMSE = oldMSE - newMSE
            changeEnergy = oldEnergy - newEnergy
            newTuple = (index, changeMSE, changeEnergy)
            newIndexMSEEnergy.append(newTuple)
            rewards[index] = changeMSE

        maxReward = max(rewards)
        indexesEnergiesOfBest = []
        for index, newMSE, newEnergy in indexMSEEnergy:
            if rewards[index] == maxReward:
                indexesEnergiesOfBest.append((index, newEnergy))
        if len(indexesEnergiesOfBest) > 0:
            indexOfBest, energyOfBest = indexesEnergiesOfBest[0]
        else:
            print('Something went wrong...')
            print(f"rewards\n {rewards}")
            print(f"indexMSEEnergy\n {indexMSEEnergy}")
            exit(0)

        for index, mse, energy in indexMSEEnergy:
            if index == indexOfBest:
                newMSE = mse
        selected['energy'] = energyOfBest
        rowToMove = selected.loc[[indexOfBest]].copy()
        unselected = gpd.GeoDataFrame(pd.concat([unselected, rowToMove], ignore_index=True), crs=unselected.crs)
        selected = selected.drop(indexOfBest).reset_index(drop=True)
        return unselected, selected, newMSE
def createMSEPlot():
    fig2, ax2 = plt.subplots()
    x = []
    y = []
    line, = ax2.plot(x, y)
    ax2.set_xlabel('Loop iteration')
    ax2.set_ylabel('MSE')
    return x, y, line, ax2, fig2


def createIGMSEPlot():
    fig2, ax2 = plt.subplots()
    ax3 = ax2.twinx()  # create a second y-axis
    x = []
    y1 = []
    y2 = []
    line1, = ax2.plot(x, y1, 'g-')  # green line for the first y-axis
    line2, = ax3.plot(x, y2, 'b-')  # blue line for the second y-axis

    ax2.set_xlabel('Loop iteration')
    ax2.set_ylabel('MSE Values', color='g')
    # ax3.set_ylabel('InfoGain Values', color='b')
    ax3.set_ylabel('Information Gainc', color='b')
    # If desired, set the color of the axis ticks to match the line color:
    ax2.tick_params(axis='y', colors='g')
    ax3.tick_params(axis='y', colors='b')

    return x, y1, y2, line1, line2, ax2, ax3, fig2

def updateMSEPlot(newX, newY, ax, fig, x, y, line):
    x.append(newX)
    y.append(newY)
    line.set_data(x, y)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()


def updateIGMSEPlot(newX, newY1, newY2, line1, line2, ax2, ax3, fig, x, y1, y2):
    x.append(newX)
    y1.append(newY1)
    y2.append(newY2)

    line1.set_data(x, y1)
    line2.set_data(x, y2)

    ax2.relim()
    ax2.autoscale_view()

    ax3.relim()
    ax3.autoscale_view()

    fig.canvas.draw()


def printResults(finalSHP, finalIteration, finalDistance, finalMSE, sensorNames):
    if len(finalSHP) == 0:
        print('No path found')
        return
    print(f"The best path's mse was {finalMSE}, and it was found on the {finalIteration}th iteration")
    print(f"It consumed {finalSHP['energy'][0]} joules and traveled {finalDistance} meters")
    sensors = getSensorNames(finalSHP['geometry'], sensorNames)
    print(f"sensors used: {sensors} ({len(sensors)})")
    print(f"best SHP:\n{finalSHP}")

def writeResults(finalSHP, finalIteration, finalDistance, finalMSE, sensorNames, outputTextName, startTime, finalMSE2):
    endTime = time.time()
    runTime = endTime - startTime
    minutes = int(runTime // 60)
    seconds = int(runTime % 60)
    if len(finalSHP) == 0:
        final_energy = 0
    else:
        final_energy = finalSHP['energy'][0]
    with open(outputTextName, mode='w', encoding='utf-8') as file:
        file.write(f"The program took {minutes} minutes and {seconds} seconds to run\n")
        file.write(f"final energy of best = {final_energy} joules\n")
        file.write(f"The best path's mse was {finalMSE}, and it was found on the {finalIteration}th iteration\n")
        file.write(f"It consumed {final_energy} joules and traveled {finalDistance} meters\n")
        sensors = getSensorNames(finalSHP['geometry'], sensorNames)
        file.write(f"sensors used: {sensors} ({len(sensors)})\n")
        file.write(f"best SHP:\n{finalSHP}")
        file.write(f"finalMSE: {finalMSE2}")

def printTime(startTime):
    endTime = time.time()
    runTime = endTime - startTime
    minutes = int(runTime // 60)
    seconds = int(runTime % 60)
    print(f"Runtime: {minutes} minutes {seconds} seconds")