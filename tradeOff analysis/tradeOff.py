from flask import Flask, request
from utils import getSubDatasets, minMaxScaling, linearCombination

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)

privacyDf = pd.read_csv(
    (os.path.join('csvs', f'average_distances.csv')), index_col=[0])
qosDf = pd.read_csv((os.path.join('csvs', f'qos.csv')), index_col=[0])


@app.route("/")
def tradeOff():
    alpha = request.args.get('alpha')
    privacySubDf = getSubDatasets(privacyDf)
    qosSubDf = getSubDatasets(qosDf)

    pValues = privacyDf["avg"]
    qosValues = pd.Series([pow(el, -1) for el in qosDf["mse"]])

    privacyScaled = minMaxScaling(pValues)
    qosScaled = minMaxScaling(qosValues)

    pSubDfValues = [df["avg"] for df in privacySubDf]
    qosSubDfValues = [pd.Series([pow(el, -1)
                                for el in df["mse"]]) for df in qosSubDf]

    privacySDScaled = [minMaxScaling(values) for values in pSubDfValues]
    qosSDScaled = [minMaxScaling(values) for values in qosSubDfValues]

    print("----------Complete Dataset----------")
    privacyQos = getTradeOff(float(alpha), privacyScaled, qosScaled, qosDf)

    privacyQosSD = []
    for ind, df1 in enumerate(privacySDScaled):
        print("----------Sub-dataset", ind+1, "----------")
        privacyQosSD.append(getTradeOff(
            float(alpha), df1, qosSDScaled[ind], qosSubDf[ind]))

    return str(privacyQos) + str(privacyQosSD)


def getTradeOff(alpha, df1, df2, mainDf):
    dfs = [pd.DataFrame([linearCombination(alpha, val, df2[ind])
                        for ind, val in enumerate(df1)], columns=["values"])]

    toRet = []

    for i, df in enumerate(dfs):
        ind, = df.index[df["values"] == np.max(df["values"])]

        toRet.append(mainDf["dumRadMin"][ind])
        toRet.append(mainDf["dumRadStep"][ind])
        toRet.append(mainDf["pertDec"][ind])

        print("Alpha: ", alpha, "\tPrivacy-QoS tradeoff: ", mainDf["dumRadMin"]
              [ind], " - ", mainDf["dumRadStep"][ind], " - ", mainDf["pertDec"][ind])

    return toRet
