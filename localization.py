import Dtw
from pandas import read_csv
from math import *

# load the dataset header=None即指明原始文件数据没有列索引，这样read_csv为自动加上列索引，除非你给定列索引的名字。
dataframe = read_csv('dataFeature.csv',engine='python',header=None)
dataset = dataframe.values
onlineData = [57.92, 58.54, 63.06, 60.30, 52.49, 51.28, 48.96, 51.89, 56.35, 59.96, 63.38]
#print(dataframe)
#print(dataset)
#print(dataset[1])


#找到数据库中最相似的一条路径
def find_max_path():
    dtw_distance = []
    for i in range(0, 8):
        val, path = Dtw.dtw(onlineData, dataset[i], Dtw.dist_for_float)
        dtw_distance.append(val)
    # 找到最相似路径的索引
    index = dtw_distance .index(min(dtw_distance))
    print(index, dataset[index])


find_max_path()