from flask import Flask, request, jsonify
from utils import getSubDatasets, minMaxScaling, linearCombination

import distutils.util
import os
import pandas as pd
import numpy as np

app = Flask(__name__)

privacyDf = pd.read_csv(
    (os.path.join('csvs', f'average_distances.csv')), index_col=[0])
qosDf = pd.read_csv((os.path.join('csvs', f'qos.csv')), index_col=[0])


@app.route("/")
def tradeOff():
    alpha = float(request.args.get('alpha'))
    dummyUpdates = bool(distutils.util.strtobool(request.args.get('dummyUp')))
    gpsPerturbated = bool(distutils.util.strtobool(
        request.args.get('gpsPert')))

    if(dummyUpdates == False and gpsPerturbated == False):
        return "No Trade-Off of Privacy-QoS to false dummyUpdates and gpsPerturbator"

    privacySub = getSubDatasets(privacyDf, dummyUpdates, gpsPerturbated)
    qosSub = getSubDatasets(qosDf, dummyUpdates, gpsPerturbated)

    privacySub["avg"] = minMaxScaling(privacySub["avg"])
    qosSub["mse"] = minMaxScaling(
        pd.Series([pow(el, -1) for el in qosSub["mse"]]))

    print("----------Trade-Off Privacy-QoS----------")
    privacyQos = getTradeOff(alpha, privacySub, qosSub)

    return privacyQos


def getTradeOff(alpha, privacy, qos):
    allTradeOff = [linearCombination(alpha, val, qos["mse"][ind])
                   for ind, val in enumerate(privacy["avg"])]

    privacy["tradeOff"] = allTradeOff
    qos["tradeOff"] = allTradeOff

    ind, = privacy.index[privacy["tradeOff"] == np.max(privacy["tradeOff"])]

    return jsonify(dummyMin=int(privacy["dumRadMin"][ind]), dummyCount=10, dummyStep=int(privacy["dumRadStep"][ind]), pertDecimals=int(privacy["pertDec"][ind]))
