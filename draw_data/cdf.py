# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from pandas import read_csv
import numpy as np
import matplotlib
from scipy import stats
import csv
import random

zhfont1 = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simsun.ttc')

# Draw 100000 samples from Normal distribution with stds of interest: samples_std1, samples_std3, samples_std10
samples_std1 = np.random.normal(50, 1, size=100)
samples_std2 = np.random.normal(30, 3, size=100)
samples_std3 = np.random.normal(30, 100, size=100)
samples_std4 = np.random.normal(30, 100, size=100)
#print(samples_std1)
#plt.show()
# Generate CDFs
x = np.linspace(0,10,35)
lab=[23,62,81,84,86,88,89,93,97,100,100,100,100,100,100,100,100,100
    , 100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100]
clm=[18,40,59,77,82,83.5,85,85.5,88,89.9,93,96,99,100,100,100,100,100
    , 100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100]
cor=[13,15,22,55,77,80,85,87,90,91,95,98,100,100,100,100,100,100
    , 100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100]
x_std1= stats.cumfreq(samples_std1)
x_std2= stats.cumfreq(samples_std2)
x_std3= stats.cumfreq(samples_std3)
x_std4= stats.cumfreq(samples_std4)
plt.plot(x,lab,label='大厅',marker='.')
plt.plot(x,cor,label='走廊',marker='+')
plt.plot(x_std3[0],label='实验室',marker='*')
plt.plot(x,clm,label='教室',marker='o')

plt.legend(loc = 'lower right')
plt.xlim(0, 10)
plt.show()

