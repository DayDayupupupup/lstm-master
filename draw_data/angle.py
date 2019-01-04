# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from pandas import read_csv
import matplotlib
import random
zhfont1 = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simsun.ttc')
gyro0,campass0 = [],[]
for i in range(100):
    gyro0.append(random.uniform(-1,1))
    campass0.append(random.uniform(-3, 3))
gyro90,campass90=[],[]
for i in range(40):
    gyro90.append(random.uniform(-1, 1))
    campass90.append(random.uniform(-3, 3))
for i in range(40):
    gyro90.append(random.uniform(-1, 1)+2*i)
    campass90.append(random.uniform(-3, 3)+2*i)
for i in range(20):
    gyro90.append(random.uniform(84, 86))
    campass90.append(random.uniform(87, 93))
gyro90fast,campass90fast=[],[]
for i in range(40):
    gyro90fast.append(random.uniform(-1, 1))
    campass90fast.append(random.uniform(-5, 5))
for i in range(10):
    gyro90fast.append(random.uniform(-2, 2)+16*i)
    campass90fast.append(random.uniform(-3, 3)+9*i)
for i in range(50):
    gyro90fast.append(random.uniform(152, 154))
    campass90fast.append(random.uniform(87, 93))
plt.ylim(-10,180)
plt.plot(gyro0,label='Gyroscope(Walking straight)', color='r',linestyle='-')
plt.plot(campass0,label='Magnetometer(Walking straight)', color='g',marker='.')
plt.plot(gyro90,label='Gyroscope(Slow turn)', color='y',linestyle='-',marker='|')
plt.plot(campass90,label='Magnetometer(Slow turn)', color='b',linestyle='--')
plt.plot(gyro90fast,label='Gyroscope(Fast turn)', color='m',linestyle='-.')
plt.plot(campass90fast,label='Magnetometer(Fast turn)', color='c',linestyle=':')
plt.legend(loc = 'upper left')
plt.xlabel("Number of sampling points")
plt.ylabel("Angle(°)")
plt.savefig('方向角.png',dpi=900)
plt.show()