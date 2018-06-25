
# coding=UTF-8


import matplotlib.pyplot as plt
import numpy as np

'''
plt.xlim(0,7)
plt.ylim(0,10)
y=np.linspace(0,10)
plt.plot(4+0*y,y,color='green',label='ground_truth')
plt.grid(True)
x1= [4,3.31,3.89,4.31,3.90,4.55,4.09,3.93,3.87,4.63]
y1 = [0,0.8,1.7,3.2,4.1,5.8,6.7,7.9,8.1,9.9]
plt.plot(x1,y1,color='red',label='ground_predict')
plt.xlabel('x')
plt.ylabel('y')

plt.show()
'''

x=[3.5,3.5,3.5,3.5,3.5,3.5,3.5,4.0,4.5,5.0,5.5]
y=[2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.0,5.0,5.0,5.0]
estx=[3.5,
3.511,
3.539,
3.562,
3.59,
3.975,
4.365,
4.753,
5.14,
5.085,
5.069,
5.009]
esty=[2.0
,2.391
,2.782
,3.172
,3.563
,3.488
,3.461
,3.408
,3.353
,2.966
,2.575
,2.188]
plt.plot(x,y,'ro',lw=2)
plt.plot(estx,esty,'bo')
plt.grid(True)
plt.show()