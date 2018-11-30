# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from pandas import read_csv
import numpy as np
import matplotlib
from scipy import stats
import csv
import random

zhfont1 = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simsun.ttc')


# Generate CDFs no kalman
'''
x = np.linspace(0,10,35)
lab=[23,62,81,84,86,88,89,93,97,100,100,100,100,100,100,100,100,100
    , 100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100]
clm=[18,40,59,77,82,83.5,85,85.5,88,89.9,93,96,99,100,100,100,100,100
    , 100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100]
cor=[13,15,22,55,77,80,85,87,90,91,95,98,100,100,100,100,100,100
    , 100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100]
hall=[7,10,20,35,38,42,45,47,48.7,51.3,55,58,62.7,66.8,70.1,73.5,76.2,80
    , 82,83,84.3,87.4,90.7,93.2,96.5,99.1,100,100,100,100,100,100,100,100,100]

plt.plot(x,hall,label='大厅',marker='.')
plt.plot(x,cor,label='走廊',marker='^')
plt.plot(x,lab,label='实验室',marker='1')
plt.plot(x,clm,label='教室',marker='<')
plt.xlabel('定位误差（m）',fontproperties=zhfont1)
plt.ylabel('CDF（%）',fontproperties=zhfont1)
plt.legend(loc = 'lower right',prop=zhfont1)
plt.xlim(0, 10)
plt.show()
'''
# Generate CDFs with kalman
'''
x = np.linspace(0,10,35)
lab=[23,62,81,84,86,89.3,90.7,93,97,100,100,100,100,100,100,100,100,100
    , 100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100]
clm=[18,40,57.2,77.8,78.3,83.5,85,85.5,88,89.9,93,96,99,100,100,100,100,100
    , 100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100]
cor=[13,16,26,55,77,80,85,87,90,91,95,98,100,100,100,100,100,100
    , 100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100]
hall=[8,12,20,34,37.5,45,47.3,52.5,58.4,64.2,68.4,74.3,77.7,80.8,83.1,84.5,87.3,91.6
    , 94.4,96.2,98,100,100,100,100,100,100,100,100,100,100,100,100,100,100]

plt.plot(x,hall,label='大厅',marker='.')
plt.plot(x,cor,label='走廊',marker='^')
plt.plot(x,lab,label='实验室',marker='1')
plt.plot(x,clm,label='教室',marker='<')
plt.xlabel('定位误差（m）',fontproperties=zhfont1)
plt.ylabel('CDF（%）',fontproperties=zhfont1)
plt.legend(loc = 'lower right',prop=zhfont1)
plt.xlim(0, 10)
plt.show()
'''
#实验室
'''
x = np.linspace(0,10,35)
lms=[0,15,23,38,56,74,88,93,97,100,100,100,100,100,100,100,100,100
    , 100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100]
particle=[0,17,28,40,54.7,73,84.3,90,91.8,92.7,93,96,99,100,100,100,100,100
    , 100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100]
pdr=[0,5,11,15,26,35,40,47,50,54.2,66.7,72.4,76,79,83.2,86.5,89,92.1
    , 94.2,96.5,98,100,100,100,100,100,100,100,100,100,100,100,100,100,100]
wifi=[0,3,7.2,11.3,14,18,22,25,31,35,38,47,50,54.6,58.4,64.3,67.8,73.6
    , 77.3,81.1,84.4,87.4,90.7,93,94,95,96,97,98,99,100,100,100,100,100]

plt.plot(x,wifi,label='WIFI',marker='.')
plt.plot(x,particle,label='粒子滤波',marker='^')
plt.plot(x,lms,label='LmsLoc',marker='1')
plt.plot(x,pdr,label='PDR',marker='<')
plt.xlabel('定位误差（m）',fontproperties=zhfont1)
plt.ylabel('CDF（%）',fontproperties=zhfont1)
plt.legend(loc = 'lower right',prop=zhfont1)
plt.xlim(0, 10)
plt.show()
'''
#dating
x = np.linspace(0,10,35)
lms=[0,13,22,35,51,68,73,88,90,95,97,98,99,100,100,100,100,100
    , 100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100]
particle=[0,14,21,35,48,55,69,82,90.5,92.7,93,94,95,97,99,100,100,100
    , 100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100]
pdr=[0,5,11,15,26,35,40,47,50,54.2,66.7,72.4,76,79,83.2,86.5,89,92.1
    , 94.2,96.5,98,100,100,100,100,100,100,100,100,100,100,100,100,100,100]
wifi=[0,2,5.2,9.3,12,16,20,22,25,30,33,42,45,47.6,48.4,54.3,57.8,63.6
    , 67.3,71.1,74.4,77.4,80.7,83,84,90,96,97,98,99,100,100,100,100,100]

plt.plot(x,wifi,label='WIFI',marker='.')
plt.plot(x,particle,label='粒子滤波',marker='^')
plt.plot(x,lms,label='LmsLoc',marker='1')
plt.plot(x,pdr,label='PDR',marker='<')
plt.xlabel('定位误差（m）',fontproperties=zhfont1)
plt.ylabel('CDF（%）',fontproperties=zhfont1)
plt.legend(loc = 'lower right',prop=zhfont1)
plt.xlim(0, 10)
plt.show()