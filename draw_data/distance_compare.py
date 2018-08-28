
import matplotlib.pyplot as plt
import numpy as np
plt.xlim(0,30)
plt.ylim(0.0,3)
x = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]

y_pdr = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

y_lstm = [0.2,  0.67 ,0.60 , 3.55 ,3.40 ,3.33 ,3.22, 3.18 , 3.05 ,2.76 ,2.54 , 2.38   ,2.32  ,2.3,
 2.4  ,2.42]
y_particle = [3.7,  3.69 ,3.65 , 3.60 ,3.57 ,3.55 ,3.48, 3.41 , 3.35 ,3.26 ,3.29 , 3.19   ,3.11  ,3.0,
 2.87  ,2.75]



plt.plot(x,y_pdr,color = 'red',label='PDR',marker='s')
plt.plot(x,y_lstm,color = 'yellow',label='LSTM',linestyle='--')
plt.plot(x,y_particle,color = 'blue',label='particle filter',linestyle='-.')
#plt.plot(x,y_MPloc,color = 'green',label='MPLoc',linestyle=':')
plt.xlabel('Distance(m)')
plt.ylabel('Time(s)')
plt.legend(['PDR','MPLoc','particle filter','MPLoc'])
plt.grid(True)

plt.show()