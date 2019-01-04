import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
np.random.seed(2)  #设置随机种子
zhfont1 = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simsun.ttc')
data = [[0.3,0.2,0.2,0.25],
        [0.6,0.5,0.5,0.6],
        [0.95,0.47,1.3,0.7],
        [1.2,0.9,2.0,0.4],
        [1.8,0.7,2.9,0.8]]

df = pd.DataFrame((data),columns=['LmsLoc', 'SmartPDR', 'MaLoc', 'WiLoc'])
df.boxplot() #也可用plot.box()
plt.ylabel('Time(s)')
plt.show()
