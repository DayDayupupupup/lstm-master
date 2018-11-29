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
plt.xlabel("采样点数目", fontproperties=zhfont1)
plt.ylabel("磁场强度(uT)", fontproperties=zhfont1)
plt.legend(['3月', '4月'],loc = 'upper right',prop=zhfont1)
plt.savefig("mag_compare.png")
plt.show()