import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
np.random.seed(2)  #设置随机种子
zhfont1 = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simsun.ttc')
data = [[0.5,0.2,0.1,0.4],
        [1.2,0.5,0.6,0.7],
        [1.7,0.8,0.9,0.95],
        [2.1,1.3,1.2,1.1],
        [2.7,1.6,1.7,1.5]]

df = pd.DataFrame((data),columns=['0.0', '0.2', '0.4', '0.6'])
df.boxplot() #也可用plot.box()
plt.xlabel('Dropout')
plt.ylabel('RMSE')
plt.show()