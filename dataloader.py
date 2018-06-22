#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2018/1/1 下午9:04
@author: Pete
@email: yuwp_1985@163.com
@file: dataloader.py
@software: PyCharm Community Edition
"""
import math
import numpy as np
import os
import pandas as pd

TIMESTAMP_BASELINE = 1490000000000

def loadAcceData(filePath, relativeTime = True):
    gravity = 9.411869  # Expect value of holding mobile phone static
    acceDF = pd.read_csv(filePath)
    acceInfo = acceDF.ix[:,['timestamp', 'acce_x', 'acce_y', 'acce_z']]
    acceTimeList = []
    acceValueList = []
    for acceRecord in acceInfo.values:
        acceTimeList.append((acceRecord[0] - TIMESTAMP_BASELINE)/ 1000.0) # milliseconds to seconds
        xAxis = acceRecord[1]
        yAxis = acceRecord[2]
        zAxis = acceRecord[3]
        acceValueList.append(math.sqrt(math.pow(xAxis, 2) + math.pow(yAxis, 2) + math.pow(zAxis, 2)) - gravity)
    if relativeTime:
        acceTimeList = [(t - acceTimeList[0]) for t in acceTimeList]
    #print(acceTimeList)
    return acceTimeList, acceValueList


def loadGyroData(filePath, relativeTime = True):
    gyroDF = pd.read_csv(filePath)
    gyroInfo = gyroDF.ix[:, ["timestamp", "gyro_z"]]
    gyroTimeList = []
    gyroValueList = []
    for gyroRecord in gyroInfo.values:
        gyroTimeList.append((gyroRecord[0] - TIMESTAMP_BASELINE) / 1000.0) # milliseconds to seconds
        gyroValueList.append(gyroRecord[1])
    if relativeTime:
        gyroTimeList = [(t - gyroTimeList[0]) for t in gyroTimeList]
    return gyroTimeList, gyroValueList


def loadCompData(filePath, relativeTime=True):
    compDF = pd.read_csv(filePath)
    compInfo = compDF.ix[:, ["timestamp", "azimut"]]
    compTimeList = []
    compValueList = []
    for compRecord in compInfo.values:
        compTimeList.append((compRecord[0] - TIMESTAMP_BASELINE) / 1000.0) # milliseconds to seconds
        compValueList.append(-1.0 * compRecord[1])    # There is a minus between compass data and gyroscope data
    if relativeTime:
        compTimeList = [(t - compTimeList[0]) for t in compTimeList]
    return compTimeList, compValueList






















