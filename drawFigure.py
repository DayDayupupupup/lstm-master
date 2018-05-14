
# coding=UTF-8


import matplotlib.pyplot as plt
import numpy as np


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
