
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
3.675,
3.777,
4.353,
4.7,
5.085,
5.6]
esty=[2.0
,2.391
,2.782
,3.172
,3.863
,4.32
,4.761
,4.88
,5.33
,5.22
,5.1]
plt.plot(x,y,'ro',lw=2,label="real")
plt.plot(estx,esty,'bo',label='estimated')
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
plt.show()