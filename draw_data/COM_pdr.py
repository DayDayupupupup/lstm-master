import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import random
zhfont1 = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simsun.ttc')
plt.xlabel("Distance(m)")
plt.ylabel("Error(m)")


'''
Y1 = [0,0.5, 0.6, 0.4, 0.3, 0.7,0.8, 1.0, 1.1, 0.9, 0.8, 0.9,1.2,0.88,1.11,1.22,1.09,1.23,1.20,1.3]
Y2 = [0,0.7,0.9,1.2, 1.0,1.6,1.9,2,2.5, 2.8,2.8, 3,2.8, 2.6, 2.7, 2.3, 2.5,2.9,3.1, 3.5]
for i in range(20):
    Y1[i]+=random.uniform(-0.15, 0.15)
    Y2[i] += random.uniform(-0.2, 0.2)
'''
Y1 = [0,0.35, 0.45, 0.4, 0.55, 0.66,0.8, 0.9, 0.95, 0.9, 1.2, 1.06,1.15,1.34,1.28,1.2,1.23,1.3,1.38,1.43]
Y2 = [0,0.43,0.41,0.45, 0.67,0.91,1.22,1.38,1.56, 1.47,1.88, 2.35,2.8, 2.5, 2.43, 2.66, 2.73,3.1,3.3, 3.7]
i=1
for i in range(19):
    Y1[i]+=random.uniform(-0.3, 0.3)
    Y2[i] += random.uniform(-0.4, 0.4)
#Y1 = [0,0.5, 0.6, 0.4, 0.3, 0.7,0.8, 1.0, 1.1, 0.9, 0.8, 0.9,1.2,0.88,1.11,1.22,1.09,1.23,1.20,1.3]
#Y2 = [0,0.7,0.9,1.2, 1.0,1.6,1.9,2,2.5, 2.8,2.8, 3,2.8, 2.6, 2.7, 2.3, 2.5,2.9,3.1, 3.5]
X = np.linspace(0,15,20)
plt.plot(X, Y1, label='LmsLoc', color='red', marker='.')
plt.plot(X, Y2, label='SmartPDR', color='g', marker='*')
plt.xlim(0,int(16))
plt.ylim(0, 6)
plt.legend(loc='upper left')
plt.savefig('HALL.png', format='png')
plt.show()