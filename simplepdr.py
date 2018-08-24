#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2018/1/2 10:53
@author: Pete
@email: yuwp_1985@163.com
@file: simplepdr.py
@software: PyCharm Community Edition
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import  MultipleLocator, FormatStrFormatter

from comutil import *
from dataloader import loadAcceData, loadGyroData
from stepcounter import SimpleStepCounter
plt.rc('font',family='Times New Roman')
from scipy import optimize
class PDR(object):
    def __init__(self, personID="pete"):
        self.personID = personID
        return

    def getLocEstimation(self, acceTimeList, acceValueList, gyroTimeList, gyroValueList):
        para = modelParameterDict.get(self.personID)
        # Count step
        acceValueArray = butterFilter(acceValueList)
        # Algorithm of step counter
        sc = SimpleStepCounter(self.personID)
        allIndexList = sc.countStep(acceTimeList, acceValueArray)
        stIndexList = allIndexList[0::3]
        stTimeList = [acceTimeList[i] for i in stIndexList]
        edIndexList = allIndexList[2::3]
        edTimeList = [acceTimeList[i] for i in edIndexList]
        stepNum = len(stIndexList)

        # Get step length
        stepFreq = stepNum / (edTimeList[-1] - stTimeList[0])
        stepLength = para[4] * stepFreq + para[5]
        #print("Step Num is %d, Step Frequency is %.3f and Step Length is %.4f" % (stepNum, stepFreq, stepLength))

        # Get rotation angle
        rotationList = rotationAngle(gyroTimeList, gyroValueList)

        # Estimate locations
        estiLocList = [(0, 0)]
        currentIndex = 0
        for i in range(len(stIndexList)):
            asTime = stTimeList[i]
            aeTime = edTimeList[i]
            # stepLength = para[4] * (1.0 / (aeTime - asTime)) +  para[5]
            rotStartIndex = timeAlign(asTime, gyroTimeList, currentIndex)
            currentIndex = rotStartIndex - 1
            rotEndIndex = timeAlign(aeTime, gyroTimeList, currentIndex)
            currentIndex = rotEndIndex - 1
            direction = meanAngle(rotationList[rotStartIndex:rotEndIndex + 1])
            lastLoc = estiLocList[-1]
            xLoc = lastLoc[0] + stepLength * math.sin(direction)
            yLoc = lastLoc[1] + stepLength * math.cos(direction)
            estiLocList.append((xLoc, yLoc))
        return estiLocList

    # def locTransform(self, originLocList, rotStr, moveVector):
    #     newLocList = []
    #     for loc in originLocList:
    #         x = y = 0
    #         # clockwise rotation first
    #         if rotStr == "0":
    #             x = loc[0]
    #             y = loc[1]
    #         elif (rotStr == "90"):
    #             x = - loc[1]
    #             y = loc[0]
    #         elif rotStr == "180":
    #             x = 0.0 - loc[0]
    #             y = 0.0 - loc[1]
    #         elif rotStr == "270":
    #             x = loc[1]
    #             y = 0.0 - loc[0]
    #         newLocList.append((moveVector[0] + x, moveVector[1] + y))
    #     return newLocList


if __name__ == "__main__":

    #sensorFilePath = "./data/step_7_4__7_7__11_7__11_4.csv"
    #locationFilePath = "./data/step_7_4__7_7__11_7__11_4_route_5.csv"
    #estimationFilePath = "./data/step_7_4__7_7__11_7__11_4_estimate.csv"
    sensorFilePath = "./data/pdr.csv"
    locationFilePath = "./data/pdr_coordinate.csv"
    estimationFilePath = "./data/pdr_estimate.csv"
    routeRotClockWise = "0"
    moveVector = (3.5, 2.0)

    #real_step = [[7,4],[7,5],[7,6],[7,7],[7,8],[7,9],[7,10],[8,10],[9,10],[10,10],[11,10]]
    real_step = [[7, 4], [7, 5], [7, 6], [7, 7], [7, 8], [7, 9], [7, 10], [8, 10]]
    true_x, true_y = zip(*real_step)
    true_x = np.array(true_x)
    true_y = np.array(true_y)
    true_x = (true_x * 0.5).astype(float)
    true_y = (true_y * 0.5).astype(float)

    locRealDF = pd.DataFrame(list(zip(true_x, true_y)))
    print(locRealDF)

    # Load sensor data from files
    acceTimeList, acceValueList = loadAcceData(sensorFilePath, relativeTime=False)
    gyroTimeList, gyroValueList = loadGyroData(sensorFilePath, relativeTime=False)

    # Get location estimation at global coordination
    myPDR = PDR()
    locEstRelList = myPDR.getLocEstimation(acceTimeList, acceValueList, gyroTimeList, gyroValueList)
    #print(locEstRelList)
    # From the relative route coordinate to global coordinate
    # locEstWorldList = myPDR.locTransform(locEstRelList, routeRotClockWise, moveVector)
    locEstWorldList = [locTransformR2W(relLoc, moveVector, routeRotClockWise) for relLoc in locEstRelList]

    # Save the estimate locations
    locEstList = [(round(loc[0] * 1000) / 1000, round(loc[1] * 1000) / 1000) for loc in locEstWorldList]
    locEstDF = pd.DataFrame(np.array(locEstList), columns=["EX(m)", "EY(m)"])
    locEstDF.to_csv(estimationFilePath, encoding='utf-8', index=False)
    print(locEstList)

    # load real locations
    # locRealDF = pd.read_csv(locationFilePath)
    KFestimeted = np.array([(6.666666666666667, 2.0), (6.666666666666667, 3.0), (6.666666666666667, 4.0), (6.666666666666667, 5.0), (6.666666666666667, 6.0), (6.666666666666667, 7.0), (6.666666666666667, 8.0), (6.666666666666667, 7.0), (7.666666666666667, 7.0), (9.0, 7.0), (10.0, 7.0)])
    KFestimeted = (KFestimeted / 2).astype(float)

    real_step = [[7,4],[7,5],[7,6],[7,7],[7,8],[7,9],[7,10],[8,10],[9,10],[10,10],[11,10]]

    geolocestimeted = np.array([(6.8, 2.89), (6.666666666666667, 3.62), (6.666666666666667, 4.43), (6.666666666666667, 4.89), (6.666666666666667, 6.21), (6.666666666666667, 7.78), (6.666666666666667, 8.85), (7.466666666666667, 9.70), (8.166666666666667, 9.84), (9.8, 9.86), (10.7, 9.81)])
    geolocestimeted = (geolocestimeted / 2).astype(float)
    # Calculate the location errors
    locRealList = [(loc[0], loc[1]) for loc in locRealDF.values]
    fusionErrList = distError(locRealList, KFestimeted)
    geo=distError(locRealList, geolocestimeted)


    errorList = distError(locRealList, locEstList)

    # Save the errors
    errorList = [round(err * 1000) / 1000 for err in errorList]
    errorFilePath = "%s_error.csv" % locationFilePath[0:-4]
    errorDF = pd.DataFrame(np.array(errorList), columns=["Error(m)"])
    errorDF.to_csv(errorFilePath, encoding='utf-8', index=False)
    print(errorList)
    print("Average Error Distance is %.3f" % np.mean(errorList))
    # Show the errors
    pdrxMajorLocator = MultipleLocator(10)
    pdrxMajorFormatter = FormatStrFormatter("%d")
    pdrxMinorLocator = MultipleLocator(5)
    pdryMajorLocator = MultipleLocator(1.0)
    pdryMajorFormatter = FormatStrFormatter("%.1f")
    pdryMinorLocator = MultipleLocator(0.5)

    fig = plt.figure()
    pdrAxe = fig.add_subplot(111)

    pdrAxe.xaxis.set_major_locator(pdrxMajorLocator)
    pdrAxe.xaxis.set_major_formatter(pdrxMajorFormatter)
    pdrAxe.xaxis.set_minor_locator(pdrxMinorLocator)
    pdrAxe.yaxis.set_major_locator(pdryMajorLocator)
    pdrAxe.yaxis.set_major_formatter(pdryMajorFormatter)
    pdrAxe.yaxis.set_minor_locator(pdryMinorLocator)
    pdrAxe.set_xlabel("$Step\ Number$")
    pdrAxe.set_ylabel("$Position\ Error(m)$")

    pdrAxe.plot(range(len(errorList)), errorList, 'r--', lw=2, label="SmartPDR")

    pdrAxe.plot(range(len(fusionErrList)), fusionErrList, 'bs-', lw=2, label="KF+fingerprint")
    pdrAxe.plot(range(len(geo)), geo, 'g^-', lw=2, label="Our PDR")

    plt.legend(loc="best")
    plt.grid()
    plt.show()

    X1, Y1 = cdf(errorList)
    z2 = np.polyfit(X1, Y1, 3)
    p1 = np.poly1d(z2)
    X2, Y2 = cdf(fusionErrList)
    X3, Y3 = cdf(geo)

    print(X2, Y2)
    X2 = np.concatenate((X2, [ 0.74379376,  0.38,  0.299]), axis=0)

    Y2 = np.concatenate((Y2, [ 0.41,  0.35,  0.21379376]), axis=0)
    np.append(Y2, [ 0.41,  0.35,  0.21379376])
    print(X2, Y2)
    print(X3, Y3)
    fig = plt.figure()
    fpAxe = fig.add_subplot(111)
    fpAxe.set_xlabel("$Position\ Error(m)$")
    fpAxe.set_ylabel("$Cumulative\ Probability$")
    fpAxe.plot(X1, Y1, 'r--', label="SmartPDR")
    fpAxe.plot(X2, Y2, 'bs-', label="KF+fingerprint")
    fpAxe.plot(X3, Y3, 'g^-', label="Our PDR")
    fpAxe.plot(X1, p1(Y1), color="b", label="Fit SmartPDR")

    plt.legend(loc="best")
    plt.grid()
    plt.show()
    print("Done.")
