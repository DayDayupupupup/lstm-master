# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from pandas import read_csv
import numpy as np
import matplotlib
from scipy import stats
import csv

zhfont1 = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simsun.ttc')

# Draw 100000 samples from Normal distribution with stds of interest: samples_std1, samples_std3, samples_std10
samples_std1 = np.random.normal(50, 1, size=100)
samples_std2 = np.random.normal(30, 3, size=100)
samples_std3 = np.random.normal(30, 100, size=100)
samples_std4 = np.random.normal(30, 100, size=100)
print(samples_std1)
#plt.show()
# Generate CDFs
0
plt.show()
x_std1= stats.cumfreq(samples_std1)
x_std2= stats.cumfreq(samples_std2)
x_std3= stats.cumfreq(samples_std3)
x_std4= stats.cumfreq(samples_std4)
plt.plot(x_std1[0],label='大厅',marker='.')
plt.plot(x_std2[0],label='走廊',marker='+')
plt.plot(x_std3[0],label='实验室',marker='*')
plt.plot(x_std4[0],label='教室',marker='o')
plt.xlim(0, 10)
plt.legend(loc = 'lower right',prop=zhfont1)
plt.show()

