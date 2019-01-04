# -*- coding: utf-8 -*-
from pandas import read_csv
import matplotlib.pyplot as plt
import matplotlib
data = read_csv("mag_compare.csv")
zhfont1 = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simsun.ttc')
#print(data)
dataset = data.values
print(dataset)
plt.plot(dataset)
plt.xlabel("Number of sampling points")
plt.ylabel("Magnetic field intensity(uT)")
plt.legend(['March', 'April'],loc = 'upper right')
plt.savefig("mag_compare.png")
plt.show()