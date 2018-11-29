import matplotlib.pyplot as plt
import numpy as np
plt.xlim(0,600)
x = [0,50,100,150,200,250,300,350,400,500]
plt.ylim(0,1)
nothing = [1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
y_pdr = [0.2, 0.25,0.3,0.37,0.48,0.71,0.88,1.0,1.23,1.26,1.32,1.55,1.71,1.85,1.9,2.0,2.19,2.28,2.5,2.8,2.6,2.9,3.21,3.43,3.77,3.88]
pddr = np.array(y_pdr)
pdr = pddr + 3.5
y_lstm = [3.7,  3.67 ,3.60 , 3.55 ,3.40 ,3.33 ,3.22, 3.18 , 3.05 ,2.76 ,2.54 , 2.38   ,2.32  ,2.3,
 2.4  ,2.42 , 2.37 ,2.20, 2.10  , 1.95 , 1.94 , 2.21 , 2.08 , 2.3 , 2.43 , 2.2]
y_particle = [3.7,  3.69 ,3.65 , 3.60 ,3.57 ,3.55 ,3.48, 3.41 , 3.35 ,3.26 ,3.29 , 3.19   ,3.11  ,3.0,
 2.87  ,2.75 , 2.65 ,2.43, 2.50  , 2.38 , 2.1, 2.0, 1.95 , 2.11 , 2.35 , 2.30]


plt.plot(x,nothing,color = 'red',label='PDR')
plt.plot(x,y_lstm,color = 'yellow',label='LSTM',linestyle='--')
plt.plot(x,y_particle,color = 'blue',label='particle filter',linestyle='-.')
#plt.plot(x,y_MPloc,color = 'green',label='MPLoc',linestyle=':')
plt.xlabel('Time(Minutes)')
plt.ylabel('Remaining battery')
plt.legend(['PDR','MPLoc','Particle filter','MPLoc'])
plt.grid(True)

plt.show()