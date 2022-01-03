import numpy as np


def getSubDatasets(df, dummyLoc, gpsPert):
    return df[(df["dummyLocation"] == dummyLoc) & (df["gpsPerturbated"] == gpsPert)].reset_index(drop=True)


def linearCombination(alpha, privacy, qos):
    return alpha * privacy + (1-alpha) * qos


def minMaxScaling(serie):
    min = np.min(serie)
    max = np.max(serie)

    return (serie - min) / (max - min)
