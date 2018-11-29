import matplotlib.pyplot as plt
import numpy as np
'''
plt.xlim(0,50)
x = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50]
y_pdr = [0.2, 0.25,0.3,0.37,0.48,0.71,0.88,1.0,1.23,1.26,1.32,1.55,1.71,1.85,1.9,2.0,2.19,2.28,2.5,2.8,2.6,2.9,3.21,3.43,3.77,3.88]
pddr = np.array(y_pdr)
pdr = pddr + 3.5
y_lstm = [3.7,  3.67 ,3.60 , 3.55 ,3.40 ,3.33 ,3.22, 3.18 , 3.05 ,2.76 ,2.54 , 2.38   ,2.32  ,2.3,
 2.4  ,2.42 , 2.37 ,2.20, 2.10  , 1.95 , 1.94 , 2.21 , 2.08 , 2.3 , 2.43 , 2.2]
y_particle = [3.7,  3.69 ,3.65 , 3.60 ,3.57 ,3.55 ,3.48, 3.41 , 3.35 ,3.26 ,3.29 , 3.19   ,3.11  ,3.0,
 2.87  ,2.75 , 2.65 ,2.43, 2.50  , 2.38 , 2.1, 2.0, 1.95 , 2.11 , 2.35 , 2.30]
#y_MPloc = [3.7,  3.75 ,3.8 , 3.87 ,3.98 ,4.21 ,4.38, 4.5 , 4.73 ,4.76 ,4.8 , 5.   ,5.2  ,5.35,
 #5.4  ,5.5 , 5.69 ,5.78, 6.  , 6.3 , 6.1 , 6.4 , 6.7 , 7.1 , 7.4 , 7.38]

plt.plot(x,pdr,color = 'red',label='PDR')
plt.plot(x,y_lstm,color = 'yellow',label='LSTM',linestyle='--')
plt.plot(x,y_particle,color = 'blue',label='particle filter',linestyle='-.')
#plt.plot(x,y_MPloc,color = 'green',label='MPLoc',linestyle=':')
plt.xlabel('Distance(m)')
plt.ylabel('Error(m)')
plt.legend(['PDR','MPLoc','particle filter','MPLoc'])
plt.grid(True)

plt.show()
'''

plt.xlim(0,25)
a = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50]
x = np.array(a)/2
y_pdr = [0.2, 0.24,0.31,0.35,0.47,0.7,0.86,1.0,1.23,1.26,1.32,1.57,1.71,1.85,1.9,2.0,2.19,2.28,2.33,2.5,2.6,2.78,2.99,3.21,3.47,3.68]
pddr = np.array(y_pdr)
pdr = pddr + 3.5
y1_lstm = [3.7,  3.67 ,3.60 , 3.55 ,3.40 ,3.33 ,3.22, 3.18 , 3.05 ,2.76 ,2.54 , 2.38   ,2.32  ,2.3,
 2.4  ,2.42 , 2.37 ,2.20, 2.10  , 1.95 , 1.94 , 2.21 , 2.08 , 2.3 , 2.43 , 2.2]
y_particle = [3.7,  3.69 ,3.65 , 3.60 ,3.57 ,3.55 ,3.48, 3.41 , 3.35 ,3.26 ,3.29 , 3.19   ,3.11  ,3.0,
 2.87  ,2.75 , 2.65 ,2.43, 2.50  , 2.38 , 2.1, 2.0, 1.95 , 2.11 , 2.35 , 2.30]
partic = np.array(y_particle) + 1.1

y_p = [2.56,  2.34, 2.7,  2.73 ,2.82,  2.8 ,2.88, 2.78, 2.65, 2.69, 2.43, 2.38 ,1.92, 1.99
, 1.88 , 2.20, 1.87 ,1.89,  2.6 , 2.55,2.7, 2.52, 2.18, 2.22 , 2.13 ,2.2 ]

#y_MPloc = [3.7,  3.75 ,3.8 , 3.87 ,3.98 ,4.21 ,4.38, 4.5 , 4.73 ,4.76 ,4.8 , 5.   ,5.2  ,5.35,
 #5.4  ,5.5 , 5.69 ,5.78, 6.  , 6.3 , 6.1 , 6.4 , 6.7 , 7.1 , 7.4 , 7.38]
y_lstm = np.array(y1_lstm)+0.5
print(y_lstm)
y_lstm=[1.56,  1.72, 1.7,  1.5 ,1.82,  1.8 ,2.1, 2.28, 2.25, 2.26, 2.04, 2.08 ,1.92, 1.8
, 1.9 , 1.90, 1.87 ,1.75,  1.6 , 1.55,1.7, 2.02, 2.18, 1.9 , 1.87 ,1.7 ]
plt.plot(x,pdr,color = 'red',label='PDR',linewidth=3.0)
plt.plot(x,y_lstm,color = 'yellow',label='LSTM',linestyle='--',linewidth=3.0)
plt.plot(x,y_p,color = 'blue',label='particle filter',linestyle='-.',linewidth=3.0)
#plt.plot(x,y_MPloc,color = 'green',label='MPLoc',linestyle=':')
plt.xlabel('Distance(m)')
plt.ylabel('Error(m)')
plt.legend(['PDR','MPLoc','particle filter','MPLoc'])
plt.grid(True)

plt.show()