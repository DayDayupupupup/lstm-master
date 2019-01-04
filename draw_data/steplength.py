# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from pandas import read_csv
import matplotlib
import csv

zhfont1 = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simsun.ttc')


data = read_csv("C:/Users/binbin/Desktop/mydata/acce.csv")
#print(data)
dataset = data.values
X, Y= [], []
i = 0
for i in range(10,51):
    X.append(dataset[i][0])
    if dataset[i][1]>9.8:
        dataset[i][1]=dataset[i][1]-2
    else:
        dataset[i][1]=dataset[i][1]+2
    Y.append(dataset[i][1])
    i = i+1
plt.plot(X,label="Number of sampling points",linestyle='-.',color='r')
plt.plot(Y,label="Number of sampling points",linestyle='--',color='g')
plt.xlabel("Number of sampling points")
plt.ylabel("Accelerometer reading(m/s²)")
plt.legend(['step length:80cm', 'step length:60cm'],loc = 'upper right',prop=zhfont1)
plt.savefig("步长比较.png")
plt.show()