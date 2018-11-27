# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from pandas import read_csv
import matplotlib
import csv

zhfont1 = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simsun.ttc')
'''

with open('acce_xyz.csv','r') as c:
    r = csv.reader(c)
    for row in c:
        X, Y, Z, sqrt, = [], [], [], []
        index = 0
        for i in r:
            if (index != 0):
                X.append(i[0])
                Y.append(i[1])
                Z.append(i[2])
                sqrt.append(i[7])
            index = index + 1
'''
data = read_csv("acce_xyz.csv")
#print(data)
dataset = data.values
X, Y, Z, sqrt, = [], [], [], []
i = 0
for i in range(0,206):
    X.append(dataset[i][0])
    Y.append(dataset[i][1])
    Z.append(dataset[i][2])
    sqrt.append(dataset[i][7])
    i = i+1

plt.figure(figsize=(6,7)) #整个现实图（框架）的大小
plt.subplot(411)
plt.plot(X,label='X轴加速度', color='r',linestyle='-')
plt.legend(loc = 'upper right', prop=zhfont1)
plt.ylabel("加速度计读数(m/s²)", fontproperties=zhfont1)
plt.subplot(412)
plt.plot(Y,label='Y轴加速度', color='y',linestyle='-')
plt.legend(loc = 'upper right', prop=zhfont1)
plt.ylabel("加速度计读数(m/s²)", fontproperties=zhfont1)
plt.subplot(413)
plt.plot(Z,label='Z轴加速度',color='b')
plt.legend(loc = 'upper right', prop=zhfont1)
plt.ylabel("加速度计读数(m/s²)", fontproperties=zhfont1)
plt.subplot(414)
plt.plot(sqrt,label='加速度模值', color='g')
plt.legend(loc = 'upper right', prop=zhfont1)
plt.xlabel("采样点数目", fontproperties=zhfont1)
plt.ylabel("加速度计读数(m/s²)", fontproperties=zhfont1)
plt.savefig('accexyz.png')
plt.show()