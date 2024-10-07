# epsilon squared greedy algorithm to find the optimal set of hover points to predict the other features without
# violating battery-of-drone budget
# Code Written by Evan Damron, University of Kentucky Computer Science

from mapping import findMinTravelDistance, plotPath, getSensorNames, addSensorsUniformRandom, getHoverPoints
import pandas as pd
import matplotlib.pyplot as plt
from ML import getMSE, calculateEntropy, discretizeData, getConditionalEntropy, getInformationGain, processData,normalizeData
import random
import time
import numpy as np
from algorithmHelpers import processFiles, getEnergy, getPointsInBudget, addRandomHP, remRandomHP, addBestHP, remBestHP, \
    createMSEPlot, updateMSEPlot, printResults, writeResults, printTime, createIGMSEPlot, updateIGMSEPlot, \
    addBestHybrid, remBestHybrid
import signal
import sys
import geopandas as gpd


def signal_handler(sig, frame):
    print('Ctrl+C pressed! Showing plot...')
    plt.show()  # display the plot
    sys.exit(0)  # exit the program

# fig1, ax1, HP_gdf, UHP_gdf, SHP_gdf, sensorNames, df,
def epsilonGreedy(fig1, ax1, HP_gdf, UHP_gdf, SHP_gdf, sensorNames, df,
                  numLoops, startTime, savePlots=False, pathPlotName="", msePlotName="", outputTextName="",
                  droneHeight=15, energyWeight=0, communicationRadius=70, energyBudget=40000, joulesPerMeter=10,
                  joulesPerSecond=35, dataSize=100, transferRate=9, minSet=False, generate=False, numSensors=37,
                  addToOriginal=True, exhaustive=True):
    # fig1, ax1, HP_gdf, UHP_gdf, SHP_gdf, sensorNames, df = processFiles(droneHeight, communicationRadius, minSet,
    #                                                                     generate, addToOriginal, numSensors)

    originalDF = df.copy()
    discreteDF = discretizeData(df)
    signal.signal(signal.SIGINT, signal_handler)
    pd.set_option('display.max_rows', 30)
    pd.set_option('display.max_columns', None)
    # print(discreteDF)
    # print(originalDF)
    if exhaustive:
        # x, y, line, ax2, fig2 = createMSEPlot()
        x, y1, y2, line1, line2, ax2, ax3, fig2 = createIGMSEPlot()
    else:
        x, y1, y2, line1, line2, ax2, ax3, fig2 = createIGMSEPlot()
        IGToSHP = {}

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
            if not exhaustive:
                df = discreteDF
            UHP_gdf, SHP_gdf = addBestHP(unselected=UHP_gdf, unselectedIB=pointsInBudget, selected=SHP_gdf,
                                         sensorNames=sensorNames, df=df, energyWeight=energyWeight,
                                         exhaustive=exhaustive)
        else:
            if not exhaustive:
                df = discreteDF
            UHP_gdf, SHP_gdf = remBestHP(UHP_gdf, SHP_gdf, sensorNames=sensorNames, df=df, energyWeight=energyWeight,
                                         joulesPerMeter=joulesPerMeter, joulesPerSecond=joulesPerSecond,
                                         dataSize=dataSize, transferRate=transferRate, exhaustive=exhaustive)

        UHP_gdf = UHP_gdf.reset_index(drop=True)
        SHP_gdf = SHP_gdf.reset_index(drop=True)
        rbProb = rbProb - 1 / numLoops

        features = getSensorNames(SHP_gdf['geometry'], sensorNames)
        if exhaustive:
            mse = getMSE(features, originalDF)
            if mse < minMSE:
                minMSE = mse
                bestSHP = SHP_gdf.copy()
                iterationOfBest = loopCount
        else:
            # mse = getMSE(features, originalDF)
            mse = 1
            if mse < minMSE:
                minMSE = mse

        IG = getInformationGain(features, discreteDF)
        if IG > maxIG:    # and len(SHP_gdf) > 8:
            maxIG = IG
            if not exhaustive:
                bestSHP = SHP_gdf.copy()
                iterationOfBest = loopCount
        # if not exhaustive:
        #     if IG not in IGToSHP.keys():
        #         IGToSHP[IG] = SHP_gdf.copy()
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
            # updateMSEPlot(newX=loopCount, newY=mse, ax=ax2, fig=fig2, x=x, y=y, line=line)
            updateIGMSEPlot(newX=loopCount, newY1=mse, newY2=IG,
                            line1=line1, line2=line2, ax2=ax2, ax3=ax3,
                            fig=fig2, x=x, y1=y1, y2=y2)
        else:
            print(f"current mse: {mse}, lowest mse yet: {minMSE}")
            updateIGMSEPlot(newX=loopCount, newY1=mse, newY2=IG, line1=line1, line2=line2, ax2=ax2, ax3=ax3,
                            fig=fig2, x=x, y1=y1, y2=y2)
        print(f"current Information Gain: {IG}, largest IG yet: {maxIG}")
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


def epsilonGreedyHybrid(numLoops, startTime, savePlots=False, pathPlotName="", msePlotName="", outputTextName="",
                  droneHeight=15, communicationRadius=70, energyBudget=60000, joulesPerMeter=10,
                  joulesPerSecond=35, dataSize=100, transferRate=9, generate=False, numSensors=37):
    fig1, ax1, HP_gdf, UHP_gdf, SHP_gdf, sensorNames, df = processFiles(droneHeight, communicationRadius, False,
                                                                        generate, True, numSensors)
    originalDF = df.copy()
    discreteDF = discretizeData(df)
    x, y1, y2, line1, line2, ax2, ax3, fig2 = createIGMSEPlot()

    print(f"Total number of Hoverpoints: {len(HP_gdf)}")
    print(f"Total number of sensors: {len(getSensorNames(HP_gdf['geometry'], sensorNames))}")
    print(f"EPSILON-GREEDY ALGORITHM WITH {numLoops} LOOPS AND ENERGY BUDGET {energyBudget}")
    arProb = 0  # probability of adding (0) and removing (1)
    rbProb = 1  # probability of random (1) and best (0)
    loopCount = 0
    pointsInBudget = getPointsInBudget(UHP_gdf, SHP_gdf, sensorNames, energyBudget, joulesPerMeter, joulesPerSecond,
                                       dataSize, transferRate)

    minMSE = float('inf')
    minEntropy = float('inf')
    mse = 99
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
            UHP_gdf, SHP_gdf, mse = addBestHybrid(unselected=UHP_gdf, unselectedIB=pointsInBudget, selected=SHP_gdf,
                                                sensorNames=sensorNames, originalDF=originalDF, discreteDF=discreteDF,
                                                     oldMSE=mse)

        else:
            UHP_gdf, SHP_gdf, mse = remBestHybrid(UHP_gdf, SHP_gdf, sensorNames=sensorNames, originalDF=originalDF,
                                discreteDF=discreteDF, joulesPerMeter=joulesPerMeter, joulesPerSecond=joulesPerSecond,
                                         dataSize=dataSize, transferRate=transferRate, oldMSE=mse)

        UHP_gdf = UHP_gdf.reset_index(drop=True)
        SHP_gdf = SHP_gdf.reset_index(drop=True)
        rbProb = rbProb - 1 / numLoops

        features = getSensorNames(SHP_gdf['geometry'], sensorNames)
        if randomNumber < raProb + rrProb:
            mse = getMSE(features, originalDF)
        if mse < minMSE:
            minMSE = mse
            bestSHP = SHP_gdf.copy()
            iterationOfBest = loopCount

        entropy = getConditionalEntropy(features, discreteDF)
        if entropy < minEntropy:    # and len(SHP_gdf) > 8:
            minEntropy = entropy
        if SHP_gdf.empty:
            totalEnergy = 0
        else:
            SHP_gdf, _ = getEnergy(dataSize=dataSize, joulesPerMeter=joulesPerMeter, joulesPerSecond=joulesPerSecond,
                                   selected=SHP_gdf, sensorNames=sensorNames, transferRate=transferRate)
            totalEnergy = SHP_gdf['energy'][0]

        print(f"This set of hoverpoints requires {totalEnergy} Joules out of the"
              f" {energyBudget} Joules in the drone's battery")
        print('Total Number of Hover Points Visited: ', len(SHP_gdf))

        updateIGMSEPlot(newX=loopCount, newY1=mse, newY2=entropy, line1=line1, line2=line2, ax2=ax2, ax3=ax3,
                            fig=fig2, x=x, y1=y1, y2=y2)
        print(f"current entropy: {entropy}, smallest entropy yet: {minEntropy}")
        print(f'current mse: {mse}, smallest mse yet: {minMSE}')
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
    # features = getSensorNames(bestSHP.geometry, sensorNames)
    # minMSE = getMSE(features, originalDF)
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
    print(f"EPSILON-GREEDY ALGORITHM WITH {numLoops} LOOPS AND ENERGY BUDGET {energyBudget}")
    printResults(bestSHP, iterationOfBest, distanceOfBest, minMSE, sensorNames)
    ax2.scatter(iterationOfBest, minMSE, color='red', zorder=10)
    plotPath(ax1, bestSHP)
    bestSHP.plot(ax=ax1, color='red', markersize=10, alpha=1)
    printTime(startTime)
    for key, value in sensorNames.items():
        if len(value) == 1:
            ax1.text(key.x, key.y, str(value[0]), fontsize=10, ha='center', va='bottom')
    if savePlots:
        writeResults(bestSHP, iterationOfBest, distanceOfBest, minMSE, sensorNames, outputTextName, startTime)
        fig1.savefig(pathPlotName, bbox_inches='tight')
        fig2.savefig(msePlotName, bbox_inches='tight')
    else:
        plt.show()

# rSeed = 1
# random.seed(rSeed)
# np.random.seed(rSeed)
# startTime = time.time()
# epsilonGreedy(numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=False,
#               energyBudget=50000, numSensors=50, addToOriginal=True, exhaustive=False)

# IGNumSensorsTests()
# rSeed = 1
# random.seed(rSeed)
# np.random.seed(rSeed)
# startTime = time.time()
# epsilonGreedy(numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=False,
#               energyBudget=50000, numSensors=50, addToOriginal=True, exhaustive=False)

# READY TO RUN
def originalBudgetTests():
    for i in range(3, 5):
        pathPlotName = f"Experiments/Budget Experiments3/original/EG/20k/path{i+1}"
        msePlotName = f"Experiments/Budget Experiments3/original/EG/20k/mse{i+1}"
        outputTextName = f"Experiments/Budget Experiments3/original/EG/20k/output{i+1}.txt"
        startTime = time.time()
        epsilonGreedy(numLoops=200, generate=False, startTime=startTime, energyWeight=0, savePlots=True, energyBudget=20000,
                      pathPlotName=pathPlotName, msePlotName=msePlotName, outputTextName=outputTextName)
    for i in range(3, 5):
        pathPlotName = f"Experiments/Budget Experiments3/original/EG/25k/path{i+1}"
        msePlotName = f"Experiments/Budget Experiments3/original/EG/25k/mse{i+1}"
        outputTextName = f"Experiments/Budget Experiments3/original/EG/25k/output{i+1}.txt"
        startTime = time.time()
        epsilonGreedy(numLoops=200, generate=False, startTime=startTime, energyWeight=0, savePlots=True, energyBudget=25000,
                      pathPlotName=pathPlotName, msePlotName=msePlotName, outputTextName=outputTextName)
    for i in range(3, 5):
        pathPlotName = f"Experiments/Budget Experiments3/original/EG/30k/path{i+1}"
        msePlotName = f"Experiments/Budget Experiments3/original/EG/30k/mse{i+1}"
        outputTextName = f"Experiments/Budget Experiments3/original/EG/30k/output{i+1}.txt"
        startTime = time.time()
        epsilonGreedy(numLoops=200, generate=False, startTime=startTime, energyWeight=0, savePlots=True, energyBudget=30000,
                      pathPlotName=pathPlotName, msePlotName=msePlotName, outputTextName=outputTextName)
    for i in range(3, 5):
        pathPlotName = f"Experiments/Budget Experiments3/original/EG/35k/path{i+1}"
        msePlotName = f"Experiments/Budget Experiments3/original/EG/35k/mse{i+1}"
        outputTextName = f"Experiments/Budget Experiments3/original/EG/35k/output{i+1}.txt"
        startTime = time.time()
        epsilonGreedy(numLoops=200, generate=False, startTime=startTime, energyWeight=0, savePlots=True, energyBudget=35000,
                      pathPlotName=pathPlotName, msePlotName=msePlotName, outputTextName=outputTextName)
    for i in range(3, 5):
        pathPlotName = f"Experiments/Budget Experiments3/original/EG/40k/path{i+1}"
        msePlotName = f"Experiments/Budget Experiments3/original/EG/40k/mse{i+1}"
        outputTextName = f"Experiments/Budget Experiments3/original/EG/40k/output{i+1}.txt"
        startTime = time.time()
        epsilonGreedy(numLoops=200, generate=False, startTime=startTime, energyWeight=0, savePlots=True, energyBudget=40000,
                      pathPlotName=pathPlotName, msePlotName=msePlotName, outputTextName=outputTextName)
    for i in range(3, 5):
        pathPlotName = f"Experiments/Budget Experiments3/original/EG/45k/path{i+1}"
        msePlotName = f"Experiments/Budget Experiments3/original/EG/45k/mse{i+1}"
        outputTextName = f"Experiments/Budget Experiments3/original/EG/45k/output{i+1}.txt"
        startTime = time.time()
        epsilonGreedy(numLoops=200, generate=False, startTime=startTime, energyWeight=0, savePlots=True, energyBudget=45000,
                      pathPlotName=pathPlotName, msePlotName=msePlotName, outputTextName=outputTextName)
    for i in range(3, 5):
        pathPlotName = f"Experiments/Budget Experiments3/original/EG/50k/path{i+1}"
        msePlotName = f"Experiments/Budget Experiments3/original/EG/50k/mse{i+1}"
        outputTextName = f"Experiments/Budget Experiments3/original/EG/50k/output{i+1}.txt"
        startTime = time.time()
        epsilonGreedy(numLoops=200, generate=False, startTime=startTime, energyWeight=0, savePlots=True, energyBudget=50000,
                      pathPlotName=pathPlotName, msePlotName=msePlotName, outputTextName=outputTextName)
    for i in range(3, 5):
        pathPlotName = f"Experiments/Budget Experiments3/original/EG/55k/path{i+1}"
        msePlotName = f"Experiments/Budget Experiments3/original/EG/55k/mse{i+1}"
        outputTextName = f"Experiments/Budget Experiments3/original/EG/55k/output{i+1}.txt"
        startTime = time.time()
        epsilonGreedy(numLoops=200, generate=False, startTime=startTime, energyWeight=0, savePlots=True, energyBudget=55000,
                      pathPlotName=pathPlotName, msePlotName=msePlotName, outputTextName=outputTextName)


def IGoriginalBudgetTests():
    for i in range(3, 5):
        pathPlotName = f"Experiments/Budget Experiments3/original/EGIG/20k/path{i+1}"
        msePlotName = f"Experiments/Budget Experiments3/original/EGIG/20k/mse{i+1}"
        outputTextName = f"Experiments/Budget Experiments3/original/EGIG/20k/output{i+1}.txt"
        startTime = time.time()
        epsilonGreedy(numLoops=200, generate=False, startTime=startTime, energyWeight=0, savePlots=True, energyBudget=20000,
                      pathPlotName=pathPlotName, msePlotName=msePlotName, outputTextName=outputTextName, exhaustive=False)
    for i in range(3, 5):
        pathPlotName = f"Experiments/Budget Experiments3/original/EGIG/25k/path{i+1}"
        msePlotName = f"Experiments/Budget Experiments3/original/EGIG/25k/mse{i+1}"
        outputTextName = f"Experiments/Budget Experiments3/original/EGIG/25k/output{i+1}.txt"
        startTime = time.time()
        epsilonGreedy(numLoops=200, generate=False, startTime=startTime, energyWeight=0, savePlots=True, energyBudget=25000,
                      pathPlotName=pathPlotName, msePlotName=msePlotName, outputTextName=outputTextName, exhaustive=False)
    for i in range(3, 5):
        pathPlotName = f"Experiments/Budget Experiments3/original/EGIG/30k/path{i+1}"
        msePlotName = f"Experiments/Budget Experiments3/original/EGIG/30k/mse{i+1}"
        outputTextName = f"Experiments/Budget Experiments3/original/EGIG/30k/output{i+1}.txt"
        startTime = time.time()
        epsilonGreedy(numLoops=200, generate=False, startTime=startTime, energyWeight=0, savePlots=True, energyBudget=30000,
                      pathPlotName=pathPlotName, msePlotName=msePlotName, outputTextName=outputTextName, exhaustive=False)
    for i in range(3, 5):
        pathPlotName = f"Experiments/Budget Experiments3/original/EGIG/35k/path{i+1}"
        msePlotName = f"Experiments/Budget Experiments3/original/EGIG/35k/mse{i+1}"
        outputTextName = f"Experiments/Budget Experiments3/original/EGIG/35k/output{i+1}.txt"
        startTime = time.time()
        epsilonGreedy(numLoops=200, generate=False, startTime=startTime, energyWeight=0, savePlots=True, energyBudget=35000,
                      pathPlotName=pathPlotName, msePlotName=msePlotName, outputTextName=outputTextName, exhaustive=False)
    for i in range(3, 5):
        pathPlotName = f"Experiments/Budget Experiments3/original/EGIG/40k/path{i+1}"
        msePlotName = f"Experiments/Budget Experiments3/original/EGIG/40k/mse{i+1}"
        outputTextName = f"Experiments/Budget Experiments3/original/EGIG/40k/output{i+1}.txt"
        startTime = time.time()
        epsilonGreedy(numLoops=200, generate=False, startTime=startTime, energyWeight=0, savePlots=True, energyBudget=40000,
                      pathPlotName=pathPlotName, msePlotName=msePlotName, outputTextName=outputTextName, exhaustive=False)
    for i in range(3, 5):
        pathPlotName = f"Experiments/Budget Experiments3/original/EGIG/45k/path{i+1}"
        msePlotName = f"Experiments/Budget Experiments3/original/EGIG/45k/mse{i+1}"
        outputTextName = f"Experiments/Budget Experiments3/original/EGIG/45k/output{i+1}.txt"
        startTime = time.time()
        epsilonGreedy(numLoops=200, generate=False, startTime=startTime, energyWeight=0, savePlots=True, energyBudget=45000,
                      pathPlotName=pathPlotName, msePlotName=msePlotName, outputTextName=outputTextName, exhaustive=False)
    for i in range(3, 5):
        pathPlotName = f"Experiments/Budget Experiments3/original/EGIG/50k/path{i+1}"
        msePlotName = f"Experiments/Budget Experiments3/original/EGIG/50k/mse{i+1}"
        outputTextName = f"Experiments/Budget Experiments3/original/EGIG/50k/output{i+1}.txt"
        startTime = time.time()
        epsilonGreedy(numLoops=200, generate=False, startTime=startTime, energyWeight=0, savePlots=True, energyBudget=50000,
                      pathPlotName=pathPlotName, msePlotName=msePlotName, outputTextName=outputTextName, exhaustive=False)
    for i in range(3, 5):
        pathPlotName = f"Experiments/Budget Experiments3/original/EGIG/55k/path{i+1}"
        msePlotName = f"Experiments/Budget Experiments3/original/EGIG/55k/mse{i+1}"
        outputTextName = f"Experiments/Budget Experiments3/original/EGIG/55k/output{i+1}.txt"
        startTime = time.time()
        epsilonGreedy(numLoops=200, generate=False, startTime=startTime, energyWeight=0, savePlots=True, energyBudget=55000,
                      pathPlotName=pathPlotName, msePlotName=msePlotName, outputTextName=outputTextName, exhaustive=False)

# originalBudgetTests()
# IGoriginalBudgetTests()
# Compare the differences between WiFi, Zigbee, Bluetooth, and UWB using exhaustive approach on original Map

# NEED TO RUN
def origTechTests():
    for i in range(3, 5):
        pathPlotName = f"Experiments/techs/original/EG/Zig/path{i + 1}"
        msePlotName = f"Experiments/techs/original/EG/Zig/mse{i + 1}"
        outputTextName = f"Experiments/techs/original/EG/Zig/output{i + 1}.txt"
        startTime = time.time()
        epsilonGreedy(numLoops=200, generate=False, startTime=startTime, energyWeight=0, savePlots=True,
                      energyBudget=40000, transferRate=.25, communicationRadius=50, droneHeight=9,
                      pathPlotName=pathPlotName, msePlotName=msePlotName, outputTextName=outputTextName, dataSize=10)
    for i in range(3, 5):
        pathPlotName = f"Experiments/techs/original/EG/WiFi/path{i + 1}"
        msePlotName = f"Experiments/techs/original/EG/WiFi/mse{i + 1}"
        outputTextName = f"Experiments/techs/original/EG/WiFi/output{i + 1}.txt"
        startTime = time.time()
        epsilonGreedy(numLoops=200, generate=False, startTime=startTime, energyWeight=0, savePlots=True,
                      energyBudget=40000, transferRate=9, communicationRadius=70, droneHeight=9,
                      pathPlotName=pathPlotName, msePlotName=msePlotName, outputTextName=outputTextName, dataSize=10)
    for i in range(3, 5):
        pathPlotName = f"Experiments/techs/original/EG/BT/path{i + 1}"
        msePlotName = f"Experiments/techs/original/EG/BT/mse{i + 1}"
        outputTextName = f"Experiments/techs/original/EG/BT/output{i + 1}.txt"
        startTime = time.time()
        epsilonGreedy(numLoops=200, generate=False, startTime=startTime, energyWeight=0, savePlots=True,
                      energyBudget=40000, transferRate=25, communicationRadius=10, droneHeight=9,
                      pathPlotName=pathPlotName, msePlotName=msePlotName, outputTextName=outputTextName, dataSize=10)
    for i in range(3, 5):
        pathPlotName = f"Experiments/techs/original/EG/UWB/path{i + 1}"
        msePlotName = f"Experiments/techs/original/EG/UWB/mse{i + 1}"
        outputTextName = f"Experiments/techs/original/EG/UWB/output{i + 1}.txt"
        startTime = time.time()
        epsilonGreedy(numLoops=200, generate=False, startTime=startTime, energyWeight=0, savePlots=True,
                      energyBudget=40000, transferRate=110, communicationRadius=10, droneHeight=9,
                      pathPlotName=pathPlotName, msePlotName=msePlotName, outputTextName=outputTextName, dataSize=10)

# origTechTests()

def IGorigTechTests():
    for i in range(3, 5):
        pathPlotName = f"Experiments/techs/original/EGIG/Zig/path{i + 1}"
        msePlotName = f"Experiments/techs/original/EGIG/Zig/mse{i + 1}"
        outputTextName = f"Experiments/techs/original/EGIG/Zig/output{i + 1}.txt"
        startTime = time.time()
        epsilonGreedy(numLoops=200, generate=False, startTime=startTime, energyWeight=0, savePlots=True,
                      energyBudget=40000, transferRate=.25, communicationRadius=50, droneHeight=9,
                      pathPlotName=pathPlotName, msePlotName=msePlotName, outputTextName=outputTextName,
                      dataSize=10, exhaustive=False)
        pathPlotName = f"Experiments/techs/original/EGIG/WiFi/path{i + 1}"
        msePlotName = f"Experiments/techs/original/EGIG/WiFi/mse{i + 1}"
        outputTextName = f"Experiments/techs/original/EGIG/WiFi/output{i + 1}.txt"
        startTime = time.time()
        epsilonGreedy(numLoops=200, generate=False, startTime=startTime, energyWeight=0, savePlots=True,
                      energyBudget=40000, transferRate=9, communicationRadius=70, droneHeight=9,
                      pathPlotName=pathPlotName, msePlotName=msePlotName, outputTextName=outputTextName,
                      dataSize=10, exhaustive=False)
        pathPlotName = f"Experiments/techs/original/EGIG/BT/path{i + 1}"
        msePlotName = f"Experiments/techs/original/EGIG/BT/mse{i + 1}"
        outputTextName = f"Experiments/techs/original/EGIG/BT/output{i + 1}.txt"
        startTime = time.time()
        epsilonGreedy(numLoops=200, generate=False, startTime=startTime, energyWeight=0, savePlots=True,
                      energyBudget=40000, transferRate=25, communicationRadius=10, droneHeight=9,
                      pathPlotName=pathPlotName, msePlotName=msePlotName, outputTextName=outputTextName,
                      dataSize=10, exhaustive=False)
        pathPlotName = f"Experiments/techs/original/EGIG/UWB/path{i + 1}"
        msePlotName = f"Experiments/techs/original/EGIG/UWB/mse{i + 1}"
        outputTextName = f"Experiments/techs/original/EGIG/UWB/output{i + 1}.txt"
        startTime = time.time()
        epsilonGreedy(numLoops=200, generate=False, startTime=startTime, energyWeight=0, savePlots=True,
                      energyBudget=40000, transferRate=110, communicationRadius=10, droneHeight=9,
                      pathPlotName=pathPlotName, msePlotName=msePlotName, outputTextName=outputTextName,
                      dataSize=10, exhaustive=False)

# IGorigTechTests()
# synthNumSensorsTests()
# def synthBudgetTests():
#     # for i in range(3):
#     #     rSeed = i + 1
#     #     random.seed(rSeed)
#     #     np.random.seed(rSeed)
#     #     pathPlotName = f"Experiments/Budget Experiments2/generated/EG/30k/map{rSeed}path1"
#     #     msePlotName = f"Experiments/Budget Experiments2/generated/EG/30k/map{rSeed}mse1"
#     #     outputTextName = f"Experiments/Budget Experiments2/generated/EG/30k/map{rSeed}output1.txt"
#     #     startTime = time.time()
#     #     epsilonGreedy(numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True, energyBudget=30000,
#     #                   pathPlotName=pathPlotName, msePlotName=msePlotName, outputTextName=outputTextName)
#     # for i in range(3):
#     #     rSeed = i + 1
#     #     random.seed(rSeed)
#     #     np.random.seed(rSeed)
#     #     pathPlotName = f"Experiments/Budget Experiments2/generated/EG/40k/map{rSeed}path1"
#     #     msePlotName = f"Experiments/Budget Experiments2/generated/EG/40k/map{rSeed}mse1"
#     #     outputTextName = f"Experiments/Budget Experiments2/generated/EG/40k/map{rSeed}output1.txt"
#     #     startTime = time.time()
#     #     epsilonGreedy(numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
#     #                   energyBudget=40000,
#     #                   pathPlotName=pathPlotName, msePlotName=msePlotName, outputTextName=outputTextName)
#     # for i in range(2):
#     #     rSeed = i + 2
#     #     random.seed(rSeed)
#     #     np.random.seed(rSeed)
#     #     pathPlotName = f"Experiments/Budget Experiments2/generated/EG/50k/map{rSeed}path1"
#     #     msePlotName = f"Experiments/Budget Experiments2/generated/EG/50k/map{rSeed}mse1"
#     #     outputTextName = f"Experiments/Budget Experiments2/generated/EG/50k/map{rSeed}output1.txt"
#     #     startTime = time.time()
#     #     epsilonGreedy(numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
#     #                   energyBudget=50000,
#     #                   pathPlotName=pathPlotName, msePlotName=msePlotName, outputTextName=outputTextName)
#     for i in range(2):
#         rSeed = i + 1
#         random.seed(rSeed)
#         np.random.seed(rSeed)
#         pathPlotName = f"Experiments/Budget Experiments2/generated/EG/60k/map{rSeed}path1"
#         msePlotName = f"Experiments/Budget Experiments2/generated/EG/60k/map{rSeed}mse1"
#         outputTextName = f"Experiments/Budget Experiments2/generated/EG/60k/map{rSeed}output1.txt"
#         startTime = time.time()
#         epsilonGreedy(numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
#                       energyBudget=60000,
#                       pathPlotName=pathPlotName, msePlotName=msePlotName, outputTextName=outputTextName)
#     for i in range(2):
#         rSeed = i + 1
#         random.seed(rSeed)
#         np.random.seed(rSeed)
#         pathPlotName = f"Experiments/Budget Experiments2/generated/EG/70k/map{rSeed}path1"
#         msePlotName = f"Experiments/Budget Experiments2/generated/EG/70k/map{rSeed}mse1"
#         outputTextName = f"Experiments/Budget Experiments2/generated/EG/70k/map{rSeed}output1.txt"
#         startTime = time.time()
#         epsilonGreedy(numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
#                       energyBudget=70000,
#                       pathPlotName=pathPlotName, msePlotName=msePlotName, outputTextName=outputTextName)
#     for i in range(2):
#         rSeed = i + 1
#         random.seed(rSeed)
#         np.random.seed(rSeed)
#         pathPlotName = f"Experiments/Budget Experiments2/generated/EG/80k/map{rSeed}path1"
#         msePlotName = f"Experiments/Budget Experiments2/generated/EG/80k/map{rSeed}mse1"
#         outputTextName = f"Experiments/Budget Experiments2/generated/EG/80k/map{rSeed}output1.txt"
#         startTime = time.time()
#         epsilonGreedy(numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
#                       energyBudget=80000,
#                       pathPlotName=pathPlotName, msePlotName=msePlotName, outputTextName=outputTextName)


# numSensorsTests3()
# numSensorsTests2()

# synthBudgetTests()
# synthNumSensorsTests()


# rSeed = 1
# random.seed(rSeed)
# np.random.seed(rSeed)
# startTime = time.time()
# epsilonGreedy(numLoops=100, generate=True, startTime=startTime, energyWeight=0, savePlots=False,
#               energyBudget=40000, numSensors=50, addToOriginal=True, exhaustive=False)


# NumSensorsTests40k2/EGIG3
def IGNumSensorsTests40k_2():
    rSeeds = [4]
    for rSeed in rSeeds:
        random.seed(rSeed)
        np.random.seed(rSeed)
        dataFolder = 'CAF_Sensor_Dataset_2/caf_sensors/Hourly'
        df = processData(dataFolder)
        sensorsGDF, df = addSensorsUniformRandom(df=df, numSensors=80)
        print(f'len columns = {len(df.columns)}')
        df = normalizeData(df)
        df = df[np.random.permutation(df.columns)]
        print(f'len columns = {len(df.columns)}')
        for j in range(0, 1):
            croppedDF = df.iloc[:, :20]
            croppedSensorsGDF = sensorsGDF[sensorsGDF['Location'].isin(croppedDF.columns)]
            fig1, ax1 = plt.subplots(figsize=(8, 8))
            ax1.set_xlim(-100, 1300 + 100)
            ax1.set_ylim(-100, 800 + 100)
            HP_gdf, sensorNames = getHoverPoints(croppedSensorsGDF, commRadius=70, ax=ax1, height=15)
            SHP_gdf = gpd.GeoDataFrame()
            SHP_gdf['geometry'] = None
            SHP_gdf = SHP_gdf.set_geometry('geometry')
            UHP_gdf = HP_gdf.copy()
            SHP_gdf.crs = 'EPSG:3857'  # pseudo-mercator
            UHP_gdf.crs = 'EPSG:3857'
            startTime = time.time()
            pathPlotName = f"Experiments/numSensorsTests40k2/EGIG3/20s/map{rSeed}path{j+1}"
            msePlotName = f"Experiments/numSensorsTests40k2/EGIG3/20s/map{rSeed}mse{j+1}"
            outputTextName = f"Experiments/numSensorsTests40k2/EGIG3/20s/map{rSeed}output{j+1}.txt"
            epsilonGreedy(fig1, ax1, HP_gdf, UHP_gdf, SHP_gdf, sensorNames, croppedDF,
                          numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
                          energyBudget=40000, pathPlotName=pathPlotName, msePlotName=msePlotName,
                          outputTextName=outputTextName, numSensors=20, addToOriginal=True, exhaustive=False)

            croppedDF = df.iloc[:, :30]
            croppedSensorsGDF = sensorsGDF[sensorsGDF['Location'].isin(croppedDF.columns)]
            fig1, ax1 = plt.subplots(figsize=(8, 8))
            ax1.set_xlim(-100, 1300 + 100)
            ax1.set_ylim(-100, 800 + 100)
            HP_gdf, sensorNames = getHoverPoints(croppedSensorsGDF, commRadius=70, ax=ax1, height=15)
            SHP_gdf = gpd.GeoDataFrame()
            SHP_gdf['geometry'] = None
            SHP_gdf = SHP_gdf.set_geometry('geometry')
            UHP_gdf = HP_gdf.copy()
            SHP_gdf.crs = 'EPSG:3857'  # pseudo-mercator
            UHP_gdf.crs = 'EPSG:3857'
            startTime = time.time()
            pathPlotName = f"Experiments/numSensorsTests40k2/EGIG3/30s/map{rSeed}path{j+1}"
            msePlotName = f"Experiments/numSensorsTests40k2/EGIG3/30s/map{rSeed}mse{j+1}"
            outputTextName = f"Experiments/numSensorsTests40k2/EGIG3/30s/map{rSeed}output{j+1}.txt"
            epsilonGreedy(fig1, ax1, HP_gdf, UHP_gdf, SHP_gdf, sensorNames, croppedDF,
                          numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
                          energyBudget=40000, pathPlotName=pathPlotName, msePlotName=msePlotName,
                          outputTextName=outputTextName, numSensors=30, addToOriginal=True, exhaustive=False)

            croppedDF = df.iloc[:, :40]
            croppedSensorsGDF = sensorsGDF[sensorsGDF['Location'].isin(croppedDF.columns)]
            fig1, ax1 = plt.subplots(figsize=(8, 8))
            ax1.set_xlim(-100, 1300 + 100)
            ax1.set_ylim(-100, 800 + 100)
            HP_gdf, sensorNames = getHoverPoints(croppedSensorsGDF, commRadius=70, ax=ax1, height=15)
            SHP_gdf = gpd.GeoDataFrame()
            SHP_gdf['geometry'] = None
            SHP_gdf = SHP_gdf.set_geometry('geometry')
            UHP_gdf = HP_gdf.copy()
            SHP_gdf.crs = 'EPSG:3857'  # pseudo-mercator
            UHP_gdf.crs = 'EPSG:3857'
            startTime = time.time()
            pathPlotName = f"Experiments/numSensorsTests40k2/EGIG3/40s/map{rSeed}path{j+1}"
            msePlotName = f"Experiments/numSensorsTests40k2/EGIG3/40s/map{rSeed}mse{j+1}"
            outputTextName = f"Experiments/numSensorsTests40k2/EGIG3/40s/map{rSeed}output{j+1}.txt"
            epsilonGreedy(fig1, ax1, HP_gdf, UHP_gdf, SHP_gdf, sensorNames, croppedDF,
                          numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
                          energyBudget=40000, pathPlotName=pathPlotName, msePlotName=msePlotName,
                          outputTextName=outputTextName, numSensors=40, addToOriginal=True, exhaustive=False)

            croppedDF = df.iloc[:, :50]
            croppedSensorsGDF = sensorsGDF[sensorsGDF['Location'].isin(croppedDF.columns)]
            fig1, ax1 = plt.subplots(figsize=(8, 8))
            ax1.set_xlim(-100, 1300 + 100)
            ax1.set_ylim(-100, 800 + 100)
            HP_gdf, sensorNames = getHoverPoints(croppedSensorsGDF, commRadius=70, ax=ax1, height=15)
            SHP_gdf = gpd.GeoDataFrame()
            SHP_gdf['geometry'] = None
            SHP_gdf = SHP_gdf.set_geometry('geometry')
            UHP_gdf = HP_gdf.copy()
            SHP_gdf.crs = 'EPSG:3857'  # pseudo-mercator
            UHP_gdf.crs = 'EPSG:3857'
            startTime = time.time()
            pathPlotName = f"Experiments/numSensorsTests40k2/EGIG3/50s/map{rSeed}path{j+1}"
            msePlotName = f"Experiments/numSensorsTests40k2/EGIG3/50s/map{rSeed}mse{j+1}"
            outputTextName = f"Experiments/numSensorsTests40k2/EGIG3/50s/map{rSeed}output{j+1}.txt"
            epsilonGreedy(fig1, ax1, HP_gdf, UHP_gdf, SHP_gdf, sensorNames, croppedDF,
                          numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
                          energyBudget=40000, pathPlotName=pathPlotName, msePlotName=msePlotName,
                          outputTextName=outputTextName, numSensors=50, addToOriginal=True, exhaustive=False)

            croppedDF = df.iloc[:, :60]
            croppedSensorsGDF = sensorsGDF[sensorsGDF['Location'].isin(croppedDF.columns)]
            fig1, ax1 = plt.subplots(figsize=(8, 8))
            ax1.set_xlim(-100, 1300 + 100)
            ax1.set_ylim(-100, 800 + 100)
            HP_gdf, sensorNames = getHoverPoints(croppedSensorsGDF, commRadius=70, ax=ax1, height=15)
            SHP_gdf = gpd.GeoDataFrame()
            SHP_gdf['geometry'] = None
            SHP_gdf = SHP_gdf.set_geometry('geometry')
            UHP_gdf = HP_gdf.copy()
            SHP_gdf.crs = 'EPSG:3857'  # pseudo-mercator
            UHP_gdf.crs = 'EPSG:3857'
            startTime = time.time()
            pathPlotName = f"Experiments/numSensorsTests40k2/EGIG3/60s/map{rSeed}path{j+1}"
            msePlotName = f"Experiments/numSensorsTests40k2/EGIG3/60s/map{rSeed}mse{j+1}"
            outputTextName = f"Experiments/numSensorsTests40k2/EGIG3/60s/map{rSeed}output{j+1}.txt"
            epsilonGreedy(fig1, ax1, HP_gdf, UHP_gdf, SHP_gdf, sensorNames, croppedDF,
                          numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
                          energyBudget=40000, pathPlotName=pathPlotName, msePlotName=msePlotName,
                          outputTextName=outputTextName, numSensors=60, addToOriginal=True, exhaustive=False)

            croppedDF = df.iloc[:, :70]
            croppedSensorsGDF = sensorsGDF[sensorsGDF['Location'].isin(croppedDF.columns)]
            fig1, ax1 = plt.subplots(figsize=(8, 8))
            ax1.set_xlim(-100, 1300 + 100)
            ax1.set_ylim(-100, 800 + 100)
            HP_gdf, sensorNames = getHoverPoints(croppedSensorsGDF, commRadius=70, ax=ax1, height=15)
            SHP_gdf = gpd.GeoDataFrame()
            SHP_gdf['geometry'] = None
            SHP_gdf = SHP_gdf.set_geometry('geometry')
            UHP_gdf = HP_gdf.copy()
            SHP_gdf.crs = 'EPSG:3857'  # pseudo-mercator
            UHP_gdf.crs = 'EPSG:3857'
            startTime = time.time()
            pathPlotName = f"Experiments/numSensorsTests40k2/EGIG3/70s/map{rSeed}path{j+1}"
            msePlotName = f"Experiments/numSensorsTests40k2/EGIG3/70s/map{rSeed}mse{j+1}"
            outputTextName = f"Experiments/numSensorsTests40k2/EGIG3/70s/map{rSeed}output{j+1}.txt"
            epsilonGreedy(fig1, ax1, HP_gdf, UHP_gdf, SHP_gdf, sensorNames, croppedDF,
                          numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
                          energyBudget=40000, pathPlotName=pathPlotName, msePlotName=msePlotName,
                          outputTextName=outputTextName, numSensors=70, addToOriginal=True, exhaustive=False)

# IGNumSensorsTests40k_2()


def numSensorsTests40k():
    seeds = [42, 2, 4]
    for rSeed in seeds:
        # random.seed(rSeed)
        np.random.seed(rSeed)
        dataFolder = 'CAF_Sensor_Dataset_2/caf_sensors/Hourly'
        df = processData(dataFolder)
        sensorsGDF, df = addSensorsUniformRandom(df=df, numSensors=80)
        df = normalizeData(df)
        df = df[np.random.permutation(df.columns)]
        for j in range(1, 2):
            # croppedDF = df.iloc[:, :20]
            # croppedSensorsGDF = sensorsGDF[sensorsGDF['Location'].isin(croppedDF.columns)]
            # fig1, ax1 = plt.subplots(figsize=(8, 8))
            # ax1.set_xlim(-100, 1300 + 100)
            # ax1.set_ylim(-100, 800 + 100)
            # HP_gdf, sensorNames = getHoverPoints(croppedSensorsGDF, commRadius=70, ax=ax1, height=15)
            # SHP_gdf = gpd.GeoDataFrame()
            # SHP_gdf['geometry'] = None
            # SHP_gdf = SHP_gdf.set_geometry('geometry')
            # UHP_gdf = HP_gdf.copy()
            # SHP_gdf.crs = 'EPSG:3857'  # pseudo-mercator
            # UHP_gdf.crs = 'EPSG:3857'
            # startTime = time.time()
            # pathPlotName = f"Experiments/numSensorsTests40k2/EG3/20s/map{rSeed}path{j + 1}"
            # msePlotName = f"Experiments/numSensorsTests40k2/EG3/20s/map{rSeed}mse{j + 1}"
            # outputTextName = f"Experiments/numSensorsTests40k2/EG3/20s/map{rSeed}output{j + 1}.txt"
            # epsilonGreedy(fig1, ax1, HP_gdf, UHP_gdf, SHP_gdf, sensorNames, croppedDF,
            #               numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
            #               energyBudget=40000, pathPlotName=pathPlotName, msePlotName=msePlotName,
            #               outputTextName=outputTextName, numSensors=20, addToOriginal=True)
            #
            # croppedDF = df.iloc[:, :30]
            # croppedSensorsGDF = sensorsGDF[sensorsGDF['Location'].isin(croppedDF.columns)]
            # fig1, ax1 = plt.subplots(figsize=(8, 8))
            # ax1.set_xlim(-100, 1300 + 100)
            # ax1.set_ylim(-100, 800 + 100)
            # HP_gdf, sensorNames = getHoverPoints(croppedSensorsGDF, commRadius=70, ax=ax1, height=15)
            # SHP_gdf = gpd.GeoDataFrame()
            # SHP_gdf['geometry'] = None
            # SHP_gdf = SHP_gdf.set_geometry('geometry')
            # UHP_gdf = HP_gdf.copy()
            # SHP_gdf.crs = 'EPSG:3857'  # pseudo-mercator
            # UHP_gdf.crs = 'EPSG:3857'
            # startTime = time.time()
            # pathPlotName = f"Experiments/numSensorsTests40k2/EG3/30s/map{rSeed}path{j + 1}"
            # msePlotName = f"Experiments/numSensorsTests40k2/EG3/30s/map{rSeed}mse{j + 1}"
            # outputTextName = f"Experiments/numSensorsTests40k2/EG3/30s/map{rSeed}output{j + 1}.txt"
            # epsilonGreedy(fig1, ax1, HP_gdf, UHP_gdf, SHP_gdf, sensorNames, croppedDF,
            #               numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
            #               energyBudget=40000, pathPlotName=pathPlotName, msePlotName=msePlotName,
            #               outputTextName=outputTextName, numSensors=30, addToOriginal=True)
            #
            # croppedDF = df.iloc[:, :40]
            # croppedSensorsGDF = sensorsGDF[sensorsGDF['Location'].isin(croppedDF.columns)]
            # fig1, ax1 = plt.subplots(figsize=(8, 8))
            # ax1.set_xlim(-100, 1300 + 100)
            # ax1.set_ylim(-100, 800 + 100)
            # HP_gdf, sensorNames = getHoverPoints(croppedSensorsGDF, commRadius=70, ax=ax1, height=15)
            # SHP_gdf = gpd.GeoDataFrame()
            # SHP_gdf['geometry'] = None
            # SHP_gdf = SHP_gdf.set_geometry('geometry')
            # UHP_gdf = HP_gdf.copy()
            # SHP_gdf.crs = 'EPSG:3857'  # pseudo-mercator
            # UHP_gdf.crs = 'EPSG:3857'
            # startTime = time.time()
            # pathPlotName = f"Experiments/numSensorsTests40k2/EG3/40s/map{rSeed}path{j + 1}"
            # msePlotName = f"Experiments/numSensorsTests40k2/EG3/40s/map{rSeed}mse{j + 1}"
            # outputTextName = f"Experiments/numSensorsTests40k2/EG3/40s/map{rSeed}output{j + 1}.txt"
            # epsilonGreedy(fig1, ax1, HP_gdf, UHP_gdf, SHP_gdf, sensorNames, croppedDF,
            #               numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
            #               energyBudget=40000, pathPlotName=pathPlotName, msePlotName=msePlotName,
            #               outputTextName=outputTextName, numSensors=40, addToOriginal=True)

            croppedDF = df.iloc[:, :70]
            croppedSensorsGDF = sensorsGDF[sensorsGDF['Location'].isin(croppedDF.columns)]
            fig1, ax1 = plt.subplots(figsize=(8, 8))
            ax1.set_xlim(-100, 1300 + 100)
            ax1.set_ylim(-100, 800 + 100)
            HP_gdf, sensorNames = getHoverPoints(croppedSensorsGDF, commRadius=70, ax=ax1, height=15)
            SHP_gdf = gpd.GeoDataFrame()
            SHP_gdf['geometry'] = None
            SHP_gdf = SHP_gdf.set_geometry('geometry')
            UHP_gdf = HP_gdf.copy()
            SHP_gdf.crs = 'EPSG:3857'  # pseudo-mercator
            UHP_gdf.crs = 'EPSG:3857'
            startTime = time.time()
            pathPlotName = f"Experiments/EG/70s/map{rSeed}path{j + 1}"
            msePlotName = f"Experiments/EG/70s/map{rSeed}mse{j + 1}"
            outputTextName = f"Experiments/EG/70s/map{rSeed}output{j + 1}.txt"
            epsilonGreedy(fig1, ax1, HP_gdf, UHP_gdf, SHP_gdf, sensorNames, croppedDF,
                          numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
                          energyBudget=40000, pathPlotName=pathPlotName, msePlotName=msePlotName,
                          outputTextName=outputTextName, numSensors=70, addToOriginal=True)

numSensorsTests40k()
# 15 bins
def IGNumSensorsTests40k_3():
    for j in range(2):
        for i in range(1, 5, 2):
            rSeed = i
            random.seed(rSeed)
            np.random.seed(rSeed)
            startTime = time.time()
            pathPlotName = f"Experiments/numSensorsTests40k3/EGIG/70s/map{rSeed}path{j+1}"
            msePlotName = f"Experiments/numSensorsTests40k3/EGIG/70s/map{rSeed}mse{j+1}"
            outputTextName = f"Experiments/numSensorsTests40k3/EGIG/70s/map{rSeed}output{j+1}.txt"
            epsilonGreedy(numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
                          energyBudget=40000, pathPlotName=pathPlotName, msePlotName=msePlotName,
                          outputTextName=outputTextName, numSensors=70, addToOriginal=True, exhaustive=False)
        for i in range(0, 5, 2):
            rSeed = i
            random.seed(rSeed)
            np.random.seed(rSeed)
            startTime = time.time()
            pathPlotName = f"Experiments/numSensorsTests40k3/EGIG/20s/map{rSeed}path{j+1}"
            msePlotName = f"Experiments/numSensorsTests40k3/EGIG/20s/map{rSeed}mse{j+1}"
            outputTextName = f"Experiments/numSensorsTests40k3/EGIG/20s/map{rSeed}output{j+1}.txt"
            epsilonGreedy(numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
                          energyBudget=40000, pathPlotName=pathPlotName, msePlotName=msePlotName,
                          outputTextName=outputTextName, numSensors=20, addToOriginal=True, exhaustive=False)
        for i in range(0, 5, 2):
            rSeed = i
            random.seed(rSeed)
            np.random.seed(rSeed)
            startTime = time.time()
            pathPlotName = f"Experiments/numSensorsTests40k3/EGIG/30s/map{rSeed}path{j+1}"
            msePlotName = f"Experiments/numSensorsTests40k3/EGIG/30s/map{rSeed}mse{j+1}"
            outputTextName = f"Experiments/numSensorsTests40k3/EGIG/30s/map{rSeed}output{j+1}.txt"
            epsilonGreedy(numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
                          energyBudget=40000, pathPlotName=pathPlotName, msePlotName=msePlotName,
                          outputTextName=outputTextName, numSensors=30, addToOriginal=True, exhaustive=False)
        for i in range(0, 5, 2):
            rSeed = i
            random.seed(rSeed)
            np.random.seed(rSeed)
            startTime = time.time()
            pathPlotName = f"Experiments/numSensorsTests40k3/EGIG/40s/map{rSeed}path{j+1}"
            msePlotName = f"Experiments/numSensorsTests40k3/EGIG/40s/map{rSeed}mse{j+1}"
            outputTextName = f"Experiments/numSensorsTests40k3/EGIG/40s/map{rSeed}output{j+1}.txt"
            epsilonGreedy(numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
                          energyBudget=40000, pathPlotName=pathPlotName, msePlotName=msePlotName,
                          outputTextName=outputTextName, numSensors=40, addToOriginal=True, exhaustive=False)
        for i in range(0, 5, 2):
            rSeed = i
            random.seed(rSeed)
            np.random.seed(rSeed)
            startTime = time.time()
            pathPlotName = f"Experiments/numSensorsTests40k3/EGIG/50s/map{rSeed}path{j+1}"
            msePlotName = f"Experiments/numSensorsTests40k3/EGIG/50s/map{rSeed}mse{j+1}"
            outputTextName = f"Experiments/numSensorsTests40k3/EGIG/50s/map{rSeed}output{j+1}.txt"
            epsilonGreedy(numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
                          energyBudget=40000, pathPlotName=pathPlotName, msePlotName=msePlotName,
                          outputTextName=outputTextName, numSensors=50, addToOriginal=True, exhaustive=False)
        for i in range(0, 5, 2):
            rSeed = i
            random.seed(rSeed)
            np.random.seed(rSeed)
            startTime = time.time()
            pathPlotName = f"Experiments/numSensorsTests40k3/EGIG/60s/map{rSeed}path{j+1}"
            msePlotName = f"Experiments/numSensorsTests40k3/EGIG/60s/map{rSeed}mse{j+1}"
            outputTextName = f"Experiments/numSensorsTests40k3/EGIG/60s/map{rSeed}output{j+1}.txt"
            epsilonGreedy(numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
                          energyBudget=40000, pathPlotName=pathPlotName, msePlotName=msePlotName,
                          outputTextName=outputTextName, numSensors=60, addToOriginal=True, exhaustive=False)

# IGNumSensorsTests40k_3()

def hybridNumSensorsTests40k():
    # for j in range(2):
    for i in range(2,5,2):
        rSeed = i
        random.seed(rSeed)
        np.random.seed(rSeed)
        startTime = time.time()
        pathPlotName = f"Experiments/numSensorsTests40k/hybrid/20s/map{rSeed}path2"
        msePlotName = f"Experiments/numSensorsTests40k/hybrid/20s/map{rSeed}mse2"
        outputTextName = f"Experiments/numSensorsTests40k/hybrid/20s/map{rSeed}output2.txt"
        epsilonGreedyHybrid(numLoops=200, generate=True, startTime=startTime, savePlots=True,
                            pathPlotName=pathPlotName, msePlotName=msePlotName, outputTextName=outputTextName,
                            energyBudget=40000, numSensors=20)
    for i in range(0,5,2):
        if i == 0:
            rSeed = 3
        else:
            rSeed = i
        random.seed(rSeed)
        np.random.seed(rSeed)
        startTime = time.time()
        pathPlotName = f"Experiments/numSensorsTests40k/hybrid/30s/map{rSeed}path2"
        msePlotName = f"Experiments/numSensorsTests40k/hybrid/30s/map{rSeed}mse2"
        outputTextName = f"Experiments/numSensorsTests40k/hybrid/30s/map{rSeed}output2.txt"
        epsilonGreedyHybrid(numLoops=200, generate=True, startTime=startTime, savePlots=True,
                            pathPlotName=pathPlotName, msePlotName=msePlotName, outputTextName=outputTextName,
                            energyBudget=40000, numSensors=30)
    for i in range(0,5,2):
        rSeed = i
        random.seed(rSeed)
        np.random.seed(rSeed)
        startTime = time.time()
        pathPlotName = f"Experiments/numSensorsTests40k/hybrid/40s/map{rSeed}path2"
        msePlotName = f"Experiments/numSensorsTests40k/hybrid/40s/map{rSeed}mse2"
        outputTextName = f"Experiments/numSensorsTests40k/hybrid/40s/map{rSeed}output2.txt"
        epsilonGreedyHybrid(numLoops=200, generate=True, startTime=startTime, savePlots=True,
                            pathPlotName=pathPlotName, msePlotName=msePlotName, outputTextName=outputTextName,
                            energyBudget=40000, numSensors=40)
    for i in range(0,5,2):
        rSeed = i
        random.seed(rSeed)
        np.random.seed(rSeed)
        startTime = time.time()
        pathPlotName = f"Experiments/numSensorsTests40k/hybrid/50s/map{rSeed}path2"
        msePlotName = f"Experiments/numSensorsTests40k/hybrid/50s/map{rSeed}mse2"
        outputTextName = f"Experiments/numSensorsTests40k/hybrid/50s/map{rSeed}output2.txt"
        epsilonGreedyHybrid(numLoops=200, generate=True, startTime=startTime, savePlots=True,
                            pathPlotName=pathPlotName, msePlotName=msePlotName, outputTextName=outputTextName,
                            energyBudget=40000, numSensors=50)
    # for i in range(2,5,2):
    rSeed = 0
    random.seed(rSeed)
    np.random.seed(rSeed)
    startTime = time.time()
    pathPlotName = f"Experiments/numSensorsTests40k/hybrid/60s/map{rSeed}path2"
    msePlotName = f"Experiments/numSensorsTests40k/hybrid/60s/map{rSeed}mse2"
    outputTextName = f"Experiments/numSensorsTests40k/hybrid/60s/map{rSeed}output2.txt"
    epsilonGreedyHybrid(numLoops=200, generate=True, startTime=startTime, savePlots=True,
                        pathPlotName=pathPlotName, msePlotName=msePlotName, outputTextName=outputTextName,
                        energyBudget=40000, numSensors=60)
    for i in range(0,5,2):
        rSeed = i
        random.seed(rSeed)
        np.random.seed(rSeed)
        startTime = time.time()
        pathPlotName = f"Experiments/numSensorsTests40k/hybrid/70s/map{rSeed}path2"
        msePlotName = f"Experiments/numSensorsTests40k/hybrid/70s/map{rSeed}mse2"
        outputTextName = f"Experiments/numSensorsTests40k/hybrid/70s/map{rSeed}output2.txt"
        epsilonGreedyHybrid(numLoops=200, generate=True, startTime=startTime, savePlots=True,
                            pathPlotName=pathPlotName, msePlotName=msePlotName, outputTextName=outputTextName,
                            energyBudget=40000, numSensors=70)
# IGNumSensorsTests40k_3()
# IGoriginalBudgetTests()
# hybridNumSensorsTests40k()

def numSensorsTimeTrials():
    # rSeed = 2
    # random.seed(rSeed)
    # np.random.seed(rSeed)
    # startTime = time.time()
    # pathPlotName = f"Experiments/numSensorsTests40k/timeTrials/min10/20s/map{rSeed}path"
    # msePlotName = f"Experiments/numSensorsTests40k/timeTrials/min10/20s/map{rSeed}mse"
    # outputTextName = f"Experiments/numSensorsTests40k/timeTrials/min10/20s/map{rSeed}output.txt"
    # epsilonGreedy(numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
    #               energyBudget=40000, pathPlotName=pathPlotName, msePlotName=msePlotName,
    #               outputTextName=outputTextName, numSensors=20, addToOriginal=True, exhaustive=False)
    # rSeed = 2
    # random.seed(rSeed)
    # np.random.seed(rSeed)
    # startTime = time.time()
    # pathPlotName = f"Experiments/numSensorsTests40k/timeTrials/min10/30s/map{rSeed}path"
    # msePlotName = f"Experiments/numSensorsTests40k/timeTrials/min10/30s/map{rSeed}mse"
    # outputTextName = f"Experiments/numSensorsTests40k/timeTrials/min10/30s/map{rSeed}output.txt"
    # epsilonGreedy(numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
    #               energyBudget=40000, pathPlotName=pathPlotName, msePlotName=msePlotName,
    #               outputTextName=outputTextName, numSensors=30, addToOriginal=True, exhaustive=False)
    # rSeed = 2
    # random.seed(rSeed)
    # np.random.seed(rSeed)
    # startTime = time.time()
    # pathPlotName = f"Experiments/numSensorsTests40k/timeTrials/min10/40s/map{rSeed}path"
    # msePlotName = f"Experiments/numSensorsTests40k/timeTrials/min10/40s/map{rSeed}mse"
    # outputTextName = f"Experiments/numSensorsTests40k/timeTrials/min10/40s/map{rSeed}output.txt"
    # epsilonGreedy(numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
    #               energyBudget=40000, pathPlotName=pathPlotName, msePlotName=msePlotName,
    #               outputTextName=outputTextName, numSensors=40, addToOriginal=True, exhaustive=False)
    # rSeed = 2
    # random.seed(rSeed)
    # np.random.seed(rSeed)
    # startTime = time.time()
    # pathPlotName = f"Experiments/numSensorsTests40k/timeTrials/min10/50s/map{rSeed}path"
    # msePlotName = f"Experiments/numSensorsTests40k/timeTrials/min10/50s/map{rSeed}mse"
    # outputTextName = f"Experiments/numSensorsTests40k/timeTrials/min10/50s/map{rSeed}output.txt"
    # epsilonGreedy(numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
    #               energyBudget=40000, pathPlotName=pathPlotName, msePlotName=msePlotName,
    #               outputTextName=outputTextName, numSensors=50, addToOriginal=True, exhaustive=False)
    # rSeed = 2
    # random.seed(rSeed)
    # np.random.seed(rSeed)
    # startTime = time.time()
    # pathPlotName = f"Experiments/numSensorsTests40k/timeTrials/min10/60s/map{rSeed}path"
    # msePlotName = f"Experiments/numSensorsTests40k/timeTrials/min10/60s/map{rSeed}mse"
    # outputTextName = f"Experiments/numSensorsTests40k/timeTrials/min10/60s/map{rSeed}output.txt"
    # epsilonGreedy(numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
    #               energyBudget=40000, pathPlotName=pathPlotName, msePlotName=msePlotName,
    #               outputTextName=outputTextName, numSensors=60, addToOriginal=True, exhaustive=False)
    rSeed = 2
    random.seed(rSeed)
    np.random.seed(rSeed)
    startTime = time.time()
    pathPlotName = f"Experiments/numSensorsTests40k/timeTrials/exhaustive/20s/map{rSeed}path"
    msePlotName = f"Experiments/numSensorsTests40k/timeTrials/exhaustive/20s/map{rSeed}mse"
    outputTextName = f"Experiments/numSensorsTests40k/timeTrials/exhaustive/20s/map{rSeed}output.txt"
    epsilonGreedy(numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
                  energyBudget=40000, pathPlotName=pathPlotName, msePlotName=msePlotName,
                  outputTextName=outputTextName, numSensors=20, addToOriginal=True)
    rSeed = 2
    random.seed(rSeed)
    np.random.seed(rSeed)
    startTime = time.time()
    pathPlotName = f"Experiments/numSensorsTests40k/timeTrials/exhaustive/30s/map{rSeed}path"
    msePlotName = f"Experiments/numSensorsTests40k/timeTrials/exhaustive/30s/map{rSeed}mse"
    outputTextName = f"Experiments/numSensorsTests40k/timeTrials/exhaustive/30s/map{rSeed}output.txt"
    epsilonGreedy(numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
                  energyBudget=40000, pathPlotName=pathPlotName, msePlotName=msePlotName,
                  outputTextName=outputTextName, numSensors=30, addToOriginal=True)
    rSeed = 2
    random.seed(rSeed)
    np.random.seed(rSeed)
    startTime = time.time()
    pathPlotName = f"Experiments/numSensorsTests40k/timeTrials/exhaustive/40s/map{rSeed}path"
    msePlotName = f"Experiments/numSensorsTests40k/timeTrials/exhaustive/40s/map{rSeed}mse"
    outputTextName = f"Experiments/numSensorsTests40k/timeTrials/exhaustive/40s/map{rSeed}output.txt"
    epsilonGreedy(numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
                  energyBudget=40000, pathPlotName=pathPlotName, msePlotName=msePlotName,
                  outputTextName=outputTextName, numSensors=40, addToOriginal=True)
    rSeed = 2
    random.seed(rSeed)
    np.random.seed(rSeed)
    startTime = time.time()
    pathPlotName = f"Experiments/numSensorsTests40k/timeTrials/exhaustive/50s/map{rSeed}path"
    msePlotName = f"Experiments/numSensorsTests40k/timeTrials/exhaustive/50s/map{rSeed}mse"
    outputTextName = f"Experiments/numSensorsTests40k/timeTrials/exhaustive/50s/map{rSeed}output.txt"
    epsilonGreedy(numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
                  energyBudget=40000, pathPlotName=pathPlotName, msePlotName=msePlotName,
                  outputTextName=outputTextName, numSensors=50, addToOriginal=True)
    rSeed = 2
    random.seed(rSeed)
    np.random.seed(rSeed)
    startTime = time.time()
    pathPlotName = f"Experiments/numSensorsTests40k/timeTrials/min10/70s/map{rSeed}path"
    msePlotName = f"Experiments/numSensorsTests40k/timeTrials/min10/70s/map{rSeed}mse"
    outputTextName = f"Experiments/numSensorsTests40k/timeTrials/min10/70s/map{rSeed}output.txt"
    epsilonGreedy(numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
                  energyBudget=40000, pathPlotName=pathPlotName, msePlotName=msePlotName,
                  outputTextName=outputTextName, numSensors=70, addToOriginal=True, exhaustive=False)

# numSensorsTimeTrials()

# DONE
# originalBudgetTests()
# origTechTests()
# random.seed(2)
# np.random.seed(2)
# startTime = time.time()
# epsilonGreedy(numLoops=100, generate=False, startTime=startTime, energyWeight=0, savePlots=False,
#                           energyBudget=40000, addToOriginal=True, exhaustive=False)

def numSensorsIGCorrect():
    for j in range(0, 1):
        for i in range(0, 5, 2):
            rSeed = i
            random.seed(rSeed)
            np.random.seed(rSeed)
            startTime = time.time()
            pathPlotName = f"Experiments/numSensorsTests40k2/EGIG/40s/map{rSeed}path{j + 1}"
            msePlotName = f"Experiments/numSensorsTests40k2/EGIG/40s/map{rSeed}mse{j + 1}"
            outputTextName = f"Experiments/numSensorsTests40k2/EGIG/40s/map{rSeed}output{j + 1}.txt"
            epsilonGreedy(numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
                          energyBudget=40000, pathPlotName=pathPlotName, msePlotName=msePlotName,
                          outputTextName=outputTextName, numSensors=40, addToOriginal=True, exhaustive=False)
        for i in range(0, 5, 2):
            rSeed = i
            random.seed(rSeed)
            np.random.seed(rSeed)
            startTime = time.time()
            pathPlotName = f"Experiments/numSensorsTests40k2/EGIG/20s/map{rSeed}path{j + 1}"
            msePlotName = f"Experiments/numSensorsTests40k2/EGIG/20s/map{rSeed}mse{j + 1}"
            outputTextName = f"Experiments/numSensorsTests40k2/EGIG/20s/map{rSeed}output{j + 1}.txt"
            epsilonGreedy(numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
                          energyBudget=40000, pathPlotName=pathPlotName, msePlotName=msePlotName,
                          outputTextName=outputTextName, numSensors=20, addToOriginal=True, exhaustive=False)
        for i in range(0, 5, 2):
            if i == 0:
                rSeed = 3
            else:
                rSeed = i
            random.seed(rSeed)
            np.random.seed(rSeed)
            startTime = time.time()
            pathPlotName = f"Experiments/numSensorsTests40k2/EGIG/30s/map{rSeed}path{j + 1}"
            msePlotName = f"Experiments/numSensorsTests40k2/EGIG/30s/map{rSeed}mse{j + 1}"
            outputTextName = f"Experiments/numSensorsTests40k2/EGIG/30s/map{rSeed}output{j + 1}.txt"
            epsilonGreedy(numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
                          energyBudget=40000, pathPlotName=pathPlotName, msePlotName=msePlotName,
                          outputTextName=outputTextName, numSensors=30, addToOriginal=True, exhaustive=False)

        for i in range(0, 5, 2):
            rSeed = i
            random.seed(rSeed)
            np.random.seed(rSeed)
            startTime = time.time()
            pathPlotName = f"Experiments/numSensorsTests40k2/EGIG/50s/map{rSeed}path{j + 1}"
            msePlotName = f"Experiments/numSensorsTests40k2/EGIG/50s/map{rSeed}mse{j + 1}"
            outputTextName = f"Experiments/numSensorsTests40k2/EGIG/50s/map{rSeed}output{j + 1}.txt"
            epsilonGreedy(numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
                          energyBudget=40000, pathPlotName=pathPlotName, msePlotName=msePlotName,
                          outputTextName=outputTextName, numSensors=50, addToOriginal=True, exhaustive=False)
        for i in range(0, 5, 2):
            rSeed = i
            random.seed(rSeed)
            np.random.seed(rSeed)
            startTime = time.time()
            pathPlotName = f"Experiments/numSensorsTests40k2/EGIG/60s/map{rSeed}path{j + 1}"
            msePlotName = f"Experiments/numSensorsTests40k2/EGIG/60s/map{rSeed}mse{j + 1}"
            outputTextName = f"Experiments/numSensorsTests40k2/EGIG/60s/map{rSeed}output{j + 1}.txt"
            epsilonGreedy(numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
                          energyBudget=40000, pathPlotName=pathPlotName, msePlotName=msePlotName,
                          outputTextName=outputTextName, numSensors=60, addToOriginal=True, exhaustive=False)
        for i in range(0, 5, 2):
            rSeed = i
            random.seed(rSeed)
            np.random.seed(rSeed)
            startTime = time.time()
            pathPlotName = f"Experiments/numSensorsTests40k2/EGIG/70s/map{rSeed}path{j + 1}"
            msePlotName = f"Experiments/numSensorsTests40k2/EGIG/70s/map{rSeed}mse{j + 1}"
            outputTextName = f"Experiments/numSensorsTests40k2/EGIG/70s/map{rSeed}output{j + 1}.txt"
            epsilonGreedy(numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
                          energyBudget=40000, pathPlotName=pathPlotName, msePlotName=msePlotName,
                          outputTextName=outputTextName, numSensors=70, addToOriginal=True, exhaustive=False)

# rSeed = 0
# random.seed(rSeed)
# np.random.seed(rSeed)
# startTime = time.time()
# pathPlotName = f"Experiments/numSensorsTests40k/EGIG3/70s/map{rSeed}path{1}"
# msePlotName = f"Experiments/numSensorsTests40k/EGIG3/70s/map{rSeed}mse{1}"
# outputTextName = f"Experiments/numSensorsTests40k/EGIG3/70s/map{rSeed}output{1}.txt"
# epsilonGreedy(numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
#               energyBudget=40000, pathPlotName=pathPlotName, msePlotName=msePlotName,
#               outputTextName=outputTextName, numSensors=70, addToOriginal=True, exhaustive=False)

# startTime = time.time()
# epsilonGreedy(numLoops=200, generate=False, startTime=startTime, energyWeight=0, savePlots=False, energyBudget=40000)
# numSensorsIGCorrect()
def synthBudgetTests2():
    rSeed = 2
    random.seed(rSeed)
    np.random.seed(rSeed)
    dataFolder = 'CAF_Sensor_Dataset_2/caf_sensors/Hourly'
    df = processData(dataFolder)
    sensorsGDF, df = addSensorsUniformRandom(df=df, numSensors=40)
    print(f'len columns = {len(df.columns)}')
    df = normalizeData(df)
    df = df[np.random.permutation(df.columns)]
    print(f'len columns = {len(df.columns)}')
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    ax1.set_xlim(-100, 1300 + 100)
    ax1.set_ylim(-100, 800 + 100)
    HP_gdf, sensorNames = getHoverPoints(sensorsGDF, commRadius=70, ax=ax1, height=15)
    SHP_gdf = gpd.GeoDataFrame()
    SHP_gdf['geometry'] = None
    SHP_gdf = SHP_gdf.set_geometry('geometry')
    UHP_gdf = HP_gdf.copy()
    SHP_gdf.crs = 'EPSG:3857'  # pseudo-mercator
    UHP_gdf.crs = 'EPSG:3857'
    # for j in range(0, 5):
    j=2
    # startTime = time.time()
    # pathPlotName = f"Experiments/Budget Experiments3/generated/EG/20k/map{rSeed}path{j+1}"
    # msePlotName = f"Experiments/Budget Experiments3/generated/EG/20k/map{rSeed}mse{j+1}"
    # outputTextName = f"Experiments/Budget Experiments3/generated/EG/20k/map{rSeed}output{j+1}.txt"
    # epsilonGreedy(fig1, ax1, HP_gdf, UHP_gdf, SHP_gdf, sensorNames, df,
    #               numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
    #               energyBudget=20000, pathPlotName=pathPlotName, msePlotName=msePlotName,
    #               outputTextName=outputTextName, numSensors=40, addToOriginal=True, exhaustive=True)
    # fig1, ax1 = plt.subplots(figsize=(8, 8))
    # ax1.set_xlim(-100, 1300 + 100)
    # ax1.set_ylim(-100, 800 + 100)
    # HP_gdf, sensorNames = getHoverPoints(sensorsGDF, commRadius=70, ax=ax1, height=15)
    # startTime = time.time()
    # pathPlotName = f"Experiments/Budget Experiments3/generated/EG/25k/map{rSeed}path{j + 1}"
    # msePlotName = f"Experiments/Budget Experiments3/generated/EG/25k/map{rSeed}mse{j + 1}"
    # outputTextName = f"Experiments/Budget Experiments3/generated/EG/25k/map{rSeed}output{j + 1}.txt"
    # epsilonGreedy(fig1, ax1, HP_gdf, UHP_gdf, SHP_gdf, sensorNames, df,
    #               numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
    #               energyBudget=25000, pathPlotName=pathPlotName, msePlotName=msePlotName,
    #               outputTextName=outputTextName, numSensors=40, addToOriginal=True, exhaustive=True)
    # fig1, ax1 = plt.subplots(figsize=(8, 8))
    # ax1.set_xlim(-100, 1300 + 100)
    # ax1.set_ylim(-100, 800 + 100)
    # HP_gdf, sensorNames = getHoverPoints(sensorsGDF, commRadius=70, ax=ax1, height=15)
    # startTime = time.time()
    # pathPlotName = f"Experiments/Budget Experiments3/generated/EG/30k/map{rSeed}path{j + 1}"
    # msePlotName = f"Experiments/Budget Experiments3/generated/EG/30k/map{rSeed}mse{j + 1}"
    # outputTextName = f"Experiments/Budget Experiments3/generated/EG/30k/map{rSeed}output{j + 1}.txt"
    # epsilonGreedy(fig1, ax1, HP_gdf, UHP_gdf, SHP_gdf, sensorNames, df,
    #               numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
    #               energyBudget=30000, pathPlotName=pathPlotName, msePlotName=msePlotName,
    #               outputTextName=outputTextName, numSensors=40, addToOriginal=True, exhaustive=True)
    # fig1, ax1 = plt.subplots(figsize=(8, 8))
    # ax1.set_xlim(-100, 1300 + 100)
    # ax1.set_ylim(-100, 800 + 100)
    # HP_gdf, sensorNames = getHoverPoints(sensorsGDF, commRadius=70, ax=ax1, height=15)
    # startTime = time.time()
    # pathPlotName = f"Experiments/Budget Experiments3/generated/EG/35k/map{rSeed}path{j + 1}"
    # msePlotName = f"Experiments/Budget Experiments3/generated/EG/35k/map{rSeed}mse{j + 1}"
    # outputTextName = f"Experiments/Budget Experiments3/generated/EG/35k/map{rSeed}output{j + 1}.txt"
    # epsilonGreedy(fig1, ax1, HP_gdf, UHP_gdf, SHP_gdf, sensorNames, df,
    #               numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
    #               energyBudget=35000, pathPlotName=pathPlotName, msePlotName=msePlotName,
    #               outputTextName=outputTextName, numSensors=40, addToOriginal=True, exhaustive=True)
    # fig1, ax1 = plt.subplots(figsize=(8, 8))
    # ax1.set_xlim(-100, 1300 + 100)
    # ax1.set_ylim(-100, 800 + 100)
    # HP_gdf, sensorNames = getHoverPoints(sensorsGDF, commRadius=70, ax=ax1, height=15)
    # startTime = time.time()
    # pathPlotName = f"Experiments/Budget Experiments3/generated/EG/40k/map{rSeed}path{j + 1}"
    # msePlotName = f"Experiments/Budget Experiments3/generated/EG/40k/map{rSeed}mse{j + 1}"
    # outputTextName = f"Experiments/Budget Experiments3/generated/EG/40k/map{rSeed}output{j + 1}.txt"
    # epsilonGreedy(fig1, ax1, HP_gdf, UHP_gdf, SHP_gdf, sensorNames, df,
    #               numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
    #               energyBudget=40000, pathPlotName=pathPlotName, msePlotName=msePlotName,
    #               outputTextName=outputTextName, numSensors=40, addToOriginal=True, exhaustive=True)
    # fig1, ax1 = plt.subplots(figsize=(8, 8))
    # ax1.set_xlim(-100, 1300 + 100)
    # ax1.set_ylim(-100, 800 + 100)
    # HP_gdf, sensorNames = getHoverPoints(sensorsGDF, commRadius=70, ax=ax1, height=15)
    # startTime = time.time()
    # pathPlotName = f"Experiments/Budget Experiments3/generated/EG/45k/map{rSeed}path{j + 1}"
    # msePlotName = f"Experiments/Budget Experiments3/generated/EG/45k/map{rSeed}mse{j + 1}"
    # outputTextName = f"Experiments/Budget Experiments3/generated/EG/45k/map{rSeed}output{j + 1}.txt"
    # epsilonGreedy(fig1, ax1, HP_gdf, UHP_gdf, SHP_gdf, sensorNames, df,
    #               numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
    #               energyBudget=45000, pathPlotName=pathPlotName, msePlotName=msePlotName,
    #               outputTextName=outputTextName, numSensors=40, addToOriginal=True, exhaustive=True)
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    ax1.set_xlim(-100, 1300 + 100)
    ax1.set_ylim(-100, 800 + 100)
    HP_gdf, sensorNames = getHoverPoints(sensorsGDF, commRadius=70, ax=ax1, height=15)
    startTime = time.time()
    pathPlotName = f"Experiments/Budget Experiments3/generated/EG/50k/map{rSeed}path{j + 1}"
    msePlotName = f"Experiments/Budget Experiments3/generated/EG/50k/map{rSeed}mse{j + 1}"
    outputTextName = f"Experiments/Budget Experiments3/generated/EG/50k/map{rSeed}output{j + 1}.txt"
    epsilonGreedy(fig1, ax1, HP_gdf, UHP_gdf, SHP_gdf, sensorNames, df,
                  numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
                  energyBudget=50000, pathPlotName=pathPlotName, msePlotName=msePlotName,
                  outputTextName=outputTextName, numSensors=40, addToOriginal=True, exhaustive=True)
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    ax1.set_xlim(-100, 1300 + 100)
    ax1.set_ylim(-100, 800 + 100)
    HP_gdf, sensorNames = getHoverPoints(sensorsGDF, commRadius=70, ax=ax1, height=15)
    startTime = time.time()
    pathPlotName = f"Experiments/Budget Experiments3/generated/EG/55k/map{rSeed}path{j + 1}"
    msePlotName = f"Experiments/Budget Experiments3/generated/EG/55k/map{rSeed}mse{j + 1}"
    outputTextName = f"Experiments/Budget Experiments3/generated/EG/55k/map{rSeed}output{j + 1}.txt"
    epsilonGreedy(fig1, ax1, HP_gdf, UHP_gdf, SHP_gdf, sensorNames, df,
                  numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
                  energyBudget=55000, pathPlotName=pathPlotName, msePlotName=msePlotName,
                  outputTextName=outputTextName, numSensors=40, addToOriginal=True, exhaustive=True)
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    ax1.set_xlim(-100, 1300 + 100)
    ax1.set_ylim(-100, 800 + 100)
    HP_gdf, sensorNames = getHoverPoints(sensorsGDF, commRadius=70, ax=ax1, height=15)
    startTime = time.time()
    pathPlotName = f"Experiments/Budget Experiments3/generated/EGIG/20k/map{rSeed}path{j+1}"
    msePlotName = f"Experiments/Budget Experiments3/generated/EGIG/20k/map{rSeed}mse{j+1}"
    outputTextName = f"Experiments/Budget Experiments3/generated/EGIG/20k/map{rSeed}output{j+1}.txt"
    epsilonGreedy(fig1, ax1, HP_gdf, UHP_gdf, SHP_gdf, sensorNames, df,
                  numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
                  energyBudget=20000, pathPlotName=pathPlotName, msePlotName=msePlotName,
                  outputTextName=outputTextName, numSensors=40, addToOriginal=True, exhaustive=False)
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    ax1.set_xlim(-100, 1300 + 100)
    ax1.set_ylim(-100, 800 + 100)
    HP_gdf, sensorNames = getHoverPoints(sensorsGDF, commRadius=70, ax=ax1, height=15)
    startTime = time.time()
    pathPlotName = f"Experiments/Budget Experiments3/generated/EGIG/25k/map{rSeed}path{j + 1}"
    msePlotName = f"Experiments/Budget Experiments3/generated/EGIG/25k/map{rSeed}mse{j + 1}"
    outputTextName = f"Experiments/Budget Experiments3/generated/EGIG/25k/map{rSeed}output{j + 1}.txt"
    epsilonGreedy(fig1, ax1, HP_gdf, UHP_gdf, SHP_gdf, sensorNames, df,
                  numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
                  energyBudget=25000, pathPlotName=pathPlotName, msePlotName=msePlotName,
                  outputTextName=outputTextName, numSensors=40, addToOriginal=True, exhaustive=False)
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    ax1.set_xlim(-100, 1300 + 100)
    ax1.set_ylim(-100, 800 + 100)
    HP_gdf, sensorNames = getHoverPoints(sensorsGDF, commRadius=70, ax=ax1, height=15)
    startTime = time.time()
    pathPlotName = f"Experiments/Budget Experiments3/generated/EGIG/30k/map{rSeed}path{j + 1}"
    msePlotName = f"Experiments/Budget Experiments3/generated/EGIG/30k/map{rSeed}mse{j + 1}"
    outputTextName = f"Experiments/Budget Experiments3/generated/EGIG/30k/map{rSeed}output{j + 1}.txt"
    epsilonGreedy(fig1, ax1, HP_gdf, UHP_gdf, SHP_gdf, sensorNames, df,
                  numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
                  energyBudget=30000, pathPlotName=pathPlotName, msePlotName=msePlotName,
                  outputTextName=outputTextName, numSensors=40, addToOriginal=True, exhaustive=False)
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    ax1.set_xlim(-100, 1300 + 100)
    ax1.set_ylim(-100, 800 + 100)
    HP_gdf, sensorNames = getHoverPoints(sensorsGDF, commRadius=70, ax=ax1, height=15)
    startTime = time.time()
    pathPlotName = f"Experiments/Budget Experiments3/generated/EGIG/35k/map{rSeed}path{j + 1}"
    msePlotName = f"Experiments/Budget Experiments3/generated/EGIG/35k/map{rSeed}mse{j + 1}"
    outputTextName = f"Experiments/Budget Experiments3/generated/EGIG/35k/map{rSeed}output{j + 1}.txt"
    epsilonGreedy(fig1, ax1, HP_gdf, UHP_gdf, SHP_gdf, sensorNames, df,
                  numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
                  energyBudget=35000, pathPlotName=pathPlotName, msePlotName=msePlotName,
                  outputTextName=outputTextName, numSensors=40, addToOriginal=True, exhaustive=False)
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    ax1.set_xlim(-100, 1300 + 100)
    ax1.set_ylim(-100, 800 + 100)
    HP_gdf, sensorNames = getHoverPoints(sensorsGDF, commRadius=70, ax=ax1, height=15)
    startTime = time.time()
    pathPlotName = f"Experiments/Budget Experiments3/generated/EGIG/40k/map{rSeed}path{j + 1}"
    msePlotName = f"Experiments/Budget Experiments3/generated/EGIG/40k/map{rSeed}mse{j + 1}"
    outputTextName = f"Experiments/Budget Experiments3/generated/EGIG/40k/map{rSeed}output{j + 1}.txt"
    epsilonGreedy(fig1, ax1, HP_gdf, UHP_gdf, SHP_gdf, sensorNames, df,
                  numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
                  energyBudget=40000, pathPlotName=pathPlotName, msePlotName=msePlotName,
                  outputTextName=outputTextName, numSensors=40, addToOriginal=True, exhaustive=False)
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    ax1.set_xlim(-100, 1300 + 100)
    ax1.set_ylim(-100, 800 + 100)
    HP_gdf, sensorNames = getHoverPoints(sensorsGDF, commRadius=70, ax=ax1, height=15)
    startTime = time.time()
    pathPlotName = f"Experiments/Budget Experiments3/generated/EGIG/45k/map{rSeed}path{j + 1}"
    msePlotName = f"Experiments/Budget Experiments3/generated/EGIG/45k/map{rSeed}mse{j + 1}"
    outputTextName = f"Experiments/Budget Experiments3/generated/EGIG/45k/map{rSeed}output{j + 1}.txt"
    epsilonGreedy(fig1, ax1, HP_gdf, UHP_gdf, SHP_gdf, sensorNames, df,
                  numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
                  energyBudget=45000, pathPlotName=pathPlotName, msePlotName=msePlotName,
                  outputTextName=outputTextName, numSensors=40, addToOriginal=True, exhaustive=False)
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    ax1.set_xlim(-100, 1300 + 100)
    ax1.set_ylim(-100, 800 + 100)
    HP_gdf, sensorNames = getHoverPoints(sensorsGDF, commRadius=70, ax=ax1, height=15)
    startTime = time.time()
    pathPlotName = f"Experiments/Budget Experiments3/generated/EGIG/50k/map{rSeed}path{j + 1}"
    msePlotName = f"Experiments/Budget Experiments3/generated/EGIG/50k/map{rSeed}mse{j + 1}"
    outputTextName = f"Experiments/Budget Experiments3/generated/EGIG/50k/map{rSeed}output{j + 1}.txt"
    epsilonGreedy(fig1, ax1, HP_gdf, UHP_gdf, SHP_gdf, sensorNames, df,
                  numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
                  energyBudget=50000, pathPlotName=pathPlotName, msePlotName=msePlotName,
                  outputTextName=outputTextName, numSensors=40, addToOriginal=True, exhaustive=False)
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    ax1.set_xlim(-100, 1300 + 100)
    ax1.set_ylim(-100, 800 + 100)
    HP_gdf, sensorNames = getHoverPoints(sensorsGDF, commRadius=70, ax=ax1, height=15)
    startTime = time.time()
    pathPlotName = f"Experiments/Budget Experiments3/generated/EGIG/55k/map{rSeed}path{j + 1}"
    msePlotName = f"Experiments/Budget Experiments3/generated/EGIG/55k/map{rSeed}mse{j + 1}"
    outputTextName = f"Experiments/Budget Experiments3/generated/EGIG/55k/map{rSeed}output{j + 1}.txt"
    epsilonGreedy(fig1, ax1, HP_gdf, UHP_gdf, SHP_gdf, sensorNames, df,
                  numLoops=200, generate=True, startTime=startTime, energyWeight=0, savePlots=True,
                  energyBudget=55000, pathPlotName=pathPlotName, msePlotName=msePlotName,
                  outputTextName=outputTextName, numSensors=40, addToOriginal=True, exhaustive=False)

# synthBudgetTests2()