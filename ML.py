import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from collections import defaultdict
from sklearn.feature_selection import mutual_info_regression
from sklearn.mixture import GaussianMixture

#iterate through folder, make new data frame indexed by date and time, columns are sensors, and value is VW_30
#from january-april 2015
def processData(folder_path):
    # Create an empty DataFrame
    df = pd.DataFrame()
    dfs = []
    # Iterate over the files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path, delimiter="\t")
            dfs.append(df)

    combined_df = pd.concat(dfs)

    # Pivot the DataFrame using pivot_table()
    pivoted_df = combined_df.pivot_table(index=['Date', 'Time'], columns='Location', values='VW_30cm', aggfunc='first')
    pivoted_df.reset_index(inplace=True)
    # Sort the DataFrame by date and time, only jan to april 2015
    pivoted_df['DateTime'] = pd.to_datetime(pivoted_df['Date'] + ' ' + pivoted_df['Time'])
    pivoted_df.set_index(['DateTime'], inplace=True)
    pivoted_df.sort_values('DateTime', inplace=True)
    pivoted_df.drop('Date', axis=1, inplace=True)
    pivoted_df.drop('Time', axis=1, inplace=True)

    start_date = '2015-01-01'
    end_date = '2015-04-30'
    filtered_df = pivoted_df.loc[(pivoted_df.index >= start_date) & (pivoted_df.index <= end_date)]
    #remove columns that are >95% NaN
    nan_percentage = filtered_df.isna().sum() / len(filtered_df) * 100
    threshold = 95
    columns_to_drop = nan_percentage[nan_percentage > threshold].index
    main_df = filtered_df.drop(columns=columns_to_drop)

    # Remove rows with at least one NaN value
    main_df = main_df.dropna()
    #main_df.to_csv('mainDataFrame.txt') save data
    return main_df


#find the mean squared error of Gradient Boosting, Linear Regression, Random Forest, and Support Vector
def testModels(dataFrame):
    #split sensors into 5 groups
    sensors = dataFrame.columns.tolist()
    numSensors = len(sensors)
    numGroups = 5 #also the number of folds
    kf = KFold(n_splits=numGroups)
    groups = []

    for _, groupIndex in kf.split(sensors):
        group = [sensors[i] for i in groupIndex]
        groups.append(group)

    models = [
        MultiOutputRegressor(GradientBoostingRegressor()),
        LinearRegression(),
        RandomForestRegressor(n_estimators=100, random_state=42),
        MultiOutputRegressor(SVR(kernel='linear'))
    ]
    modelMSEValues = []
    for model in models:
        groupMSEValues = []
        #Every group is the target once, and the others are the features
        for group in groups:
            targetSensors = group
            featureSensors = [sensor for sensor in sensors if sensor not in targetSensors]
            x = dataFrame[featureSensors]
            y = dataFrame[targetSensors]

            foldMSEValues = []

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', TransformedTargetRegressor(regressor=model, transformer=StandardScaler()))
            ])
            pipeline.fit(x_train, y_train)
            predictions = pipeline.predict(x_test)
            #Calculate mean squared error of one model with one particular group as target, and one particular fold as testing data
            groupMSE = mean_squared_error(y_test, predictions)
            groupMSEValues.append(groupMSE)
        avgModelMSE = np.mean(groupMSEValues)
        modelMSEValues.append(avgModelMSE)
    print('Average MSE of Gradient Boosting Regressor: ', modelMSEValues[0])
    print('Average MSE of Linear Regression: ', modelMSEValues[1])
    print('Average MSE of Random Forest Regressor: ', modelMSEValues[2])
    print('Average MSE of Support Vector Regressor: ', modelMSEValues[3])

def selectModel():
    dataFolder = 'CAF_Sensor_Dataset_2/caf_sensors/Hourly'
    df = processData(dataFolder)
    testModels(df)

# selectModel()
#using gradient boosting regressor, get mse of given features, with nonselected features as the target
def getMSE(selectedSensors, df):
    selectedSensors.sort()
    sensors = df.columns.tolist()
    featureSensors = [sensor for sensor in selectedSensors if sensor in sensors]
    if len(featureSensors) == 0:
        return float('inf')
    targetSensors = [sensor for sensor in sensors if sensor not in featureSensors]
    if len(targetSensors) == 0:
        return 0
    x = df[featureSensors]
    y = df[targetSensors]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', TransformedTargetRegressor(
            regressor=MultiOutputRegressor(GradientBoostingRegressor(random_state=42)),
            transformer=StandardScaler()
        ))
    ])
    if len(targetSensors) == 1:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', TransformedTargetRegressor(
                regressor=GradientBoostingRegressor(random_state=42),
                transformer=StandardScaler()
            ))
        ])
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    # Calculate mean squared error of one model with one particular group as target, and one particular fold as testing data
    mse = mean_squared_error(y_test, predictions)
    return mse

def discretizeData(df, numBins=10):
    dfCopy = df.copy()
    # dfCopy = dfCopy.iloc[:1000]
    minReading = dfCopy.values.min()
    maxReading = dfCopy.values.max()
    binEdges = np.linspace(minReading, maxReading, numBins+1)
    discretizedDF = pd.DataFrame()
    for column in dfCopy.columns:
        discretizedDF[column] = pd.cut(dfCopy[column], bins=binEdges, labels=range(1, numBins + 1), include_lowest=True)
    return discretizedDF

def calculateEntropy(features, df):
    data = df[features]
    flatData = data.values.flatten()
    totalData = len(flatData)
    _, counts = np.unique(flatData, return_counts=True)
    probabilities = counts / totalData
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy


# def getInformationGain(selectedSensors, df):
#     sensors = df.columns.tolist()
#     featureSensors = [sensor for sensor in selectedSensors if sensor in sensors]
#     if len(featureSensors) == 0:
#         return float('-inf')
#     targetSensors = [sensor for sensor in sensors if sensor not in featureSensors]
#     if len(targetSensors) == 0:
#         return float('inf')
#     dfCopy = df.copy()
#     dfCopy['featuresCompound'] = dfCopy[featureSensors].apply(tuple, axis=1)
#     dfCopy['targetsCompound'] = dfCopy[targetSensors].apply(tuple, axis=1)
#
#     def entropy(data):
#         uniqueTuples, counts = np.unique(data, return_counts=True)
#         probabilities = counts / len(data)
#         return -np.sum(probabilities * np.log2(probabilities))
#
#     targetEntropy = entropy(dfCopy['targetsCompound'])
#
#     # calculate conditional entropy
#     jointProb = dfCopy.groupby(['featuresCompound', 'targetsCompound']).size() / len(dfCopy)
#     featuresProb = dfCopy.groupby('featuresCompound').size() / len(dfCopy)
#     # conditionalEntropy = sum(entropy(groaAZup['targetsCompound']) * len(group) / len(dfCopy) for _, group in df_grouped)
#     conditionalEntropy = -sum(jointProb[X, Y] * np.log2(jointProb[X, Y] / featuresProb[X]) for X, Y in jointProb.index)
#
#     infoGain = targetEntropy - conditionalEntropy
#     return infoGain
# def getInformationGain(selectedSensors, df):
#     sensors = df.columns.tolist()
#     featureSensors = [sensor for sensor in selectedSensors if sensor in sensors]
#     targetSensors = [sensor for sensor in sensors if sensor not in featureSensors]
#
#     if not featureSensors or not targetSensors:
#         return float('-inf')
#
#     def entropy(data):
#         _, counts = np.unique(data, return_counts=True)
#         probabilities = counts / len(data)
#         return -np.sum(probabilities * np.log2(probabilities))
#
#     total_infoGain = 0
#
#     for feature_sensor in featureSensors:
#         for target_sensor in targetSensors:
#             targetEntropy = entropy(df[target_sensor])
#
#             # Creating a joint dataframe to calculate conditional entropy
#             joint_df = pd.concat([df[feature_sensor], df[target_sensor]], axis=1)
#             joint_df.columns = ['Feature', 'Target']
#
#             # Conditional entropy: entropy of target given the feature
#             grouped = joint_df.groupby('Feature')
#             conditionalEntropy = sum(entropy(group['Target']) * len(group) / len(joint_df) for _, group in grouped)
#
#             infoGain = targetEntropy - conditionalEntropy
#             total_infoGain += infoGain
#
#     average_infoGain = total_infoGain / (len(featureSensors) * len(targetSensors))
#     return average_infoGain

# def getInformationGain(selectedSensors, df):
#     sensors = df.columns.tolist()
#     featureSensors = [sensor for sensor in selectedSensors if sensor in sensors]
#     if len(featureSensors) == 0:
#         return float('-inf')
#     targetSensors = [sensor for sensor in sensors if sensor not in featureSensors]
#     if len(targetSensors) == 0:
#         return float('inf')
#     dfCopy = df.copy()
#     # dfCopy['featuresCompound'] = dfCopy[featureSensors].apply(tuple, axis=1)
#     # dfCopy['targetsCompound'] = dfCopy[targetSensors].apply(tuple, axis=1)
#     totalMI = 0
#     for featureSensor in featureSensors:
#         totalFeatureMI = 0
#         for targetSensor in targetSensors:
#
#             U = np.array(dfCopy[featureSensor])
#             V = np.array(dfCopy[targetSensor])
#             jointProb = {}
#             for u in set(U):
#                 for v in set(V):
#                     jointProb[(u, v)] = np.mean(np.logical_and(U == u, V == v))
#
#             uProb = {}
#             for u in set(U):
#                 uProb[u] = np.mean(np.mean(U == u))
#
#             vProb = {}
#             for v in set(V):
#                 vProb[v] = np.mean(V == v)
#
#             for u, v in jointProb:
#                 if jointProb[(u, v)] > 0:
#                     totalFeatureMI += jointProb[(u, v)] * np.log2(jointProb[(u, v)] / (uProb[u] * vProb[v]))
#         averageFeatureMI = totalFeatureMI / (len(featureSensors) * len(targetSensors))
#         totalMI += averageFeatureMI
#     return totalMI
def getInformationGain(selectedSensors, df):
    sensors = df.columns.tolist()
    featureSensors = [sensor for sensor in selectedSensors if sensor in sensors]
    if len(featureSensors) == 0:
        return float('-inf')
    targetSensors = [sensor for sensor in sensors if sensor not in featureSensors]
    if len(targetSensors) == 0:
        return float('inf')
    dfCopy = df.copy()
    df['joint_features'] = df[featureSensors].apply(tuple, axis=1)
    df['joint_targets'] = df[targetSensors].apply(tuple, axis=1)

    # Compute joint probabilities
    joint_counts = defaultdict(float)
    total_rows = len(df)

    for _, row in df.iterrows():
        joint_counts[(row['joint_features'], row['joint_targets'])] += 1 / total_rows

    # Compute marginal probabilities
    marginal_features = defaultdict(float)
    for joint, prob in joint_counts.items():
        marginal_features[joint[0]] += prob

    marginal_targets = defaultdict(float)
    for joint, prob in joint_counts.items():
        marginal_targets[joint[1]] += prob

    # Compute mutual information, which is the better term for this
    MI = 0
    for (f, t), joint_prob in joint_counts.items():
        # P(f, t) * log(P(f, t) / (P(f) * P(t)))
        # Equivalent to H(t) - H(t|f)
        MI += joint_prob * np.log2(joint_prob / (marginal_features[f] * marginal_targets[t]))

    return MI
def getConditionalEntropy(selectedSensors, df):
    sensors = df.columns.tolist()
    featureSensors = [sensor for sensor in selectedSensors if sensor in sensors]
    if len(featureSensors) == 0:
        return float('inf')
    targetSensors = [sensor for sensor in sensors if sensor not in featureSensors]
    if len(targetSensors) == 0:
        return 0
    dfCopy = df.copy()
    dfCopy['featuresCompound'] = dfCopy[featureSensors].apply(tuple, axis=1)
    dfCopy['targetsCompound'] = dfCopy[targetSensors].apply(tuple, axis=1)
    # Calculate the entropy of targets given features
    df_grouped = dfCopy.groupby('featuresCompound')
    def entropy(data):
        _, counts = np.unique(data, return_counts=True)
        probabilities = counts / len(data)
        return -np.sum(probabilities * np.log2(probabilities))
    # Sum over each group's contribution to conditional entropy
    conditionalEntropy = sum(entropy(group['targetsCompound']) * len(group) / len(dfCopy) for _, group in df_grouped)
    return conditionalEntropy


def calculate_permutation_importance(X, y):
    # Create a GradientBoostingRegressor model
    gb_model = GradientBoostingRegressor()

    # Fit the model on the dataset
    gb_model.fit(X, y)

    # Calculate feature importance using permutation feature importance
    result = permutation_importance(gb_model, X, y, n_repeats=10, random_state=42)

    # # Print the feature importances
    # for i in result.importances_mean.argsort()[::-1]:
    #     print(f"{X.columns[i]:<8}"
    #           f"{result.importances_mean[i]:.3f}"
    #           f" +/- {result.importances_std[i]:.3f}")
    return result

def calculate_feature_importance(Dataset):
    train_data = Dataset
    num_var = len(Dataset.columns)
    print(f'len of dataset columns doing feature importance: {num_var}')
    # Initialize a dictionary for the importance stores of each var
    importance_dict={}
    counter = 0
    for var in np.arange(num_var):
        counter += 1
        print(f'{counter} / {num_var}')
        #print(f"calculating for var {var}:")
        # Create a regression dataset - column i is the prediction, the rest columns are the features
        Y_train=train_data.iloc[:, var]
        X_train=train_data.drop(train_data.columns[var], axis=1)
        result= calculate_permutation_importance(X_train, Y_train)
        # Print the feature importances
        for i in result.importances_mean.argsort()[::-1]:
            if X_train.columns[i] not in importance_dict:
                importance_score_list=[result.importances_mean[i]]
                importance_dict[X_train.columns[i]]=importance_score_list
            else:
                importance_dict[X_train.columns[i]].append(result.importances_mean[i])
    # calculate the average of the importance score and rank them in desending order
    averages_importance_score = {key: sum(values) / len(values) for key, values in importance_dict.items()}
    # print(averages_importance_score)
    # sorted_vars = sorted(averages_importance_score, key=lambda k: averages_importance_score[k], reverse=True)
    return averages_importance_score


def getSyntheticDF_two(df, numSensors, numToAverage=3):
    originalSeed = np.random.get_state()
    np.random.seed(42)
    numRows = 1000
    originalNumSensors = len(df.columns)
    if originalNumSensors % numToAverage == 0:
        print('FIX YOUR CODE DUMMY')
        exit()
    maxGenerate = (originalNumSensors // numToAverage) * numToAverage
    maxSensors = originalNumSensors + maxGenerate
    numToGenerate = numSensors - originalNumSensors
    if maxGenerate < numToGenerate:
        print(f'Cant generate that many sensors ({numToGenerate}), max sensor count = {maxSensors}, and starting with'
              f' {originalNumSensors}')
        exit()
    if numToGenerate < 0:
        df = df.sample(numSensors, axis=1)
    elif numToGenerate > 0:
        print(f'num to generate: {numToGenerate}')
        newDF = df.copy()
        numGenerated = 0
        for i in range(0, numToGenerate*3, numToAverage):
            indexes = [i%originalNumSensors, (i+1)%originalNumSensors, (i+2)%originalNumSensors]
            subset = df.iloc[:, indexes]
            newColumnName = f"synthetic {numGenerated + 1}"
            newDF[newColumnName] = subset.mean(axis=1)
            numGenerated += 1
        df = newDF
    covarianceMatrix = df.cov().values
    meanVector = df.mean().values
    syntheticArray = np.random.multivariate_normal(meanVector, covarianceMatrix, numRows)
    columns = [f"synthetic {i + 1}" for i in range(numSensors)]
    synthetic_df = pd.DataFrame(syntheticArray, columns=columns)
    np.random.set_state(originalSeed)
    return synthetic_df


def normalizeData(df):
    # print(len(df.columns))
    originalSeed = np.random.get_state()
    np.random.seed(42)
    numRows = len(df)
    covarianceMatrix = df.cov().values
    meanVector = df.mean().values
    syntheticArray = np.random.multivariate_normal(meanVector, covarianceMatrix, numRows)
    columns = df.columns
    synthetic_df = pd.DataFrame(syntheticArray, columns=columns)
    np.random.set_state(originalSeed)
    return synthetic_df



def getSyntheticDF(df):
    originalSeed = np.random.get_state()
    np.random.seed(42)
    numRows = 1000
    originalCovarianceMatrix = df.cov().values
    originalMeanVector = df.mean().values
    # find min/max covariance and variance by iterating through original covariance matrix
    minVariance = float('inf')
    maxVariance = float('-inf')
    minCovariance = float('inf')
    maxCovariance = float('-inf')
    # Iterate through the matrix
    for i in range(originalCovarianceMatrix.shape[0]):
        for j in range(originalCovarianceMatrix.shape[1]):
            value = originalCovarianceMatrix[i, j]

            if i == j:  # Diagonal element
                minVariance = min(minVariance, value)
                maxVariance = max(maxVariance, value)
            else:  # Off-diagonal element
                minCovariance = min(minCovariance, value)
                maxCovariance = max(maxCovariance, value)
    minMean = np.min(originalMeanVector)
    maxMean = np.max(originalMeanVector)
    numColumns = 100
    newMeanVector = np.random.uniform(minMean, maxMean, size=numColumns)
    A = np.random.uniform(minCovariance, maxCovariance, size=[numColumns, numColumns])
    cov = np.dot(A, A.transpose())
    syntheticArray = np.random.multivariate_normal(newMeanVector, cov, numRows)

    columns = [f"synthetic {i + 1}" for i in range(numColumns)]
    synthetic_df = pd.DataFrame(syntheticArray, columns=columns)
    np.random.set_state(originalSeed)
    return synthetic_df

if __name__ == '__main__':
    df = pd.read_csv('training_data0514.csv', index_col=0)
    print(df)
    testModels(df)
    