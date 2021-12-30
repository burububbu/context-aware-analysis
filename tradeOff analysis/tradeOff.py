from flask import Flask, request, Response, jsonify
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
    alpha = request.args.get('alpha')
    if(alpha == None):
        return Response("No alpha set", status=400)

    alpha = float(alpha)

    if(alpha < 0 or alpha > 1):
        return Response("Alpha must be a value between 0 and 1", status=400)

    dummyUpdates = bool(distutils.util.strtobool(request.args.get('dumUpd')))
    gpsPerturbated = bool(distutils.util.strtobool(
        request.args.get('gpsPert')))

    if(dummyUpdates == False and gpsPerturbated == False):
        return Response("No Trade-Off of Privacy-QoS to false dummyUpdates and gpsPerturbator", status=400)

    privacySub = getSubDatasets(privacyDf, dummyUpdates, gpsPerturbated)
    qosSub = getSubDatasets(qosDf, dummyUpdates, gpsPerturbated)

    privacySub["avg"] = minMaxScaling(privacySub["avg"])
    qosSub["mse"] = minMaxScaling([pow(el, -1) for el in qosSub["mse"]])

    privacyQos = getTradeOff(alpha, privacySub, qosSub)

    return privacyQos


def getTradeOff(alpha, privacy, qos):
    allTradeOff = [linearCombination(alpha, avg, mse)
                   for avg, mse in zip(privacy["avg"], qos["mse"])]

    privacy["tradeOff"] = allTradeOff
    # qos["tradeOff"] = allTradeOff

    tradeOff = privacy[privacy["tradeOff"] == np.max(privacy["tradeOff"])]

    return jsonify(dummyMin=int(tradeOff["dumRadMin"]), dummyCount=10, dummyStep=int(tradeOff["dumRadStep"]), pertDecimals=int(tradeOff["pertDec"]))
