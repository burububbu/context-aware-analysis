import numpy as np

def getSubDatasets(df):
    firstDf = df[(df["dummyLocation"] == False) & (
        df["gpsPerturbated"] == True)].reset_index(drop=True)
    secondDf = df[(df["dummyLocation"] == True) & (
        df["gpsPerturbated"] == False)].reset_index(drop=True)
    thirdDf = df[(df["dummyLocation"] == True) & (
        df["gpsPerturbated"] == True)].reset_index(drop=True)

    return [firstDf, secondDf, thirdDf]


def linearCombination(alpha, privacy, qos):
    return alpha * privacy + (1-alpha) * qos


def minMaxScaling(df):
    min = np.min(df)
    max = np.max(df)

    return (df - min) / (max - min)
