# -*- coding: utf-8 -*-
from pandas import read_csv
import matplotlib.pyplot as plt
import matplotlib
data = read_csv("C:/Users/binbin/Desktop/mydata/acce.csv")
#print(data)
dataset = data.values
print(dataset)
plt.plot(dataset)
zhfont1 = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simsun.ttc')
'''
plt.xlabel("Number of samples")
plt.ylabel("Acceleration(m/s²)")
plt.legend(['Xiaomi6', 'Galaxy Note3'],loc = 'upper right')
plt.savefig("acce_compare.png")
'''
print(dataset)
plt.xlabel("采样点数目", fontproperties=zhfont1)
plt.ylabel("加速度计读数(m/s²)", fontproperties=zhfont1)
plt.legend(['MI 6', 'Galaxy S8'],loc = 'upper right')
plt.legend(fontproperties=zhfont1)
plt.savefig("加速度比较.png")
plt.show()

