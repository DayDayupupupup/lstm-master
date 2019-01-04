# encoding=utf-8
import matplotlib.pyplot as plt
import random
import matplotlib
zhfont1 = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simsun.ttc')
x = [0.1,1.1,2.2,3.4,4.3,5.4]
x1=[]
x2=[]
x3 = []
for i in x:
    if i!=0.1:
        p=i-random.random()
        q=i-random.random()
        v=i-random.random()
    else:
        p=i
        q=i
        v=i
    x1.append(p)
    x2.append(v)
    x3.append(q)
y=[1.0,0.8,0.6,0.4,0.2,0]


plt.plot(x, y, color='green', ms=10,label='SmartPDR',linestyle='--')
plt.plot(x2, y, color='red', ms=10,label='LmsLoc',linestyle='-.')
plt.plot(x3, y, color='blue', ms=10,label='Maloc',marker='.')
plt.plot(x1, y, color='black',label='WiLoc',marker='*')

plt.legend()  # 让图例生效
# plt.xticks(x1, name, rotation=45)
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel('Time(h)') #X轴标签
plt.ylabel("Remaining battery(%)") #Y轴标签


plt.show()
