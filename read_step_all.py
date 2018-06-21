import csv
import numpy as np
import pandas as pd
from sys import argv

try:
    OFFSET = float(argv[1])
except:
    OFFSET = 15

def load_step(file_name = 'data_3_3__3_10.csv', file_path = 'data/', grid = [[7,0],[7,1],[7,2],[7,3],[7,4],[7,5],[7,6],[7,7],[7,8],[7,9],[7,10],[7,11],[7,12]]):
    """
    load step info, return generated data as step1.txt
    """
    
    # f = file("step2.txt","w+")
    # 20 21 22 23 24 25 26 27 37 47 57 67 77
    # grid = [[2,0],[2,1],[2,2],[2,3],[2,4],[2,5],[2,6],[2,7],[3,7],[4,7],[5,7],[6,7],[7,7]]
    # grid = [[13,0],[13,1],[13,2],[13,3],[13,4],[13,5],[13,6],[13,7],[13,8],[13,9],[13,10],[13,11],[13,12],[13,13]]
    # grid = [[13,0],[13,1],[13,2],[13,3],[13,4],[13,5],[13,6],[13,7],[13,8],[13,9]]
    # grid = [[13,0],[13,1],[13,2],[13,3],[13,4],[13,5]]
    # grid = [[3,3],[3,4],[3,5],[3,6],[3,7],[3,8],[3,9],[3,10]]
    # grid = [[3,3],[4,3],[5,3],[6,3],[7,3],[8,3],[9,3]]

    # grid = [[0,0],[0,1],[0,2],[0,3],[0,4],[0,5],[0,6],[0,7],[0,8],[0,9],[0,10],[0,11],[0,12]]

    # grid = [[7,4],[7,5],[7,6],[7,7],[7,8],[7,9],[7,10]]
    # grid = [[7,4],[7,5],[7,6],[7,7],[7,8],[7,9],[7,10],[8,10],[9,10],[10,10],[11,10]]
    # grid = [[7,4],[7,5],[7,6],[7,7],[7,8],[7,9],[7,10],[7,11]]
    # grid = [[7,4],[7,5],[7,6],[7,7],[7,8],[7,9],[7,10],[7,11],[7,12]]
    # grid = [[7,4],[7,5],[7,6],[7,7],[7,8],[7,9],[7,10],[7,11],[7,12],[7,13]]
    # grid =  [[7,4],[7,5],[7,6],[7,7],[7,8],[7,9],[7,10],[7,11],[7,12],[7,13],[7,14]]
    # grid = [[7,4],[7,5],[7,6],[7,7],[7,8],[7,9],[7,10],[7,11],[7,12],[7,13],[7,14],[7,15]]
    # grid = [[7,4],[7,5],[7,6],[7,7],[7,8],[7,9],[7,10],[7,11],[7,12],[7,13],[7,14],[7,15],[7,16]]
    # grid = [[7,4],[7,5],[7,6],[7,7],[8,7],[9,7],[10,7],[11,7],[11,6],[11,5],[11,4]]
    # grid = [[7,4],[7,5],[7,6],[7,7],[7,8],[8,8],[9,8],[10,8],[11,8]]
    # grid = [[7,4],[7,5],[7,6],[7,7],[7,8],[7,9]]

    file_line_write = []
    file_write = []
    line_split = []
    df = pd.read_csv(file_path+file_name, encoding='utf8', skiprows=0)
    orientation = df['orientation_x']+OFFSET ##########offset of orientation#########
    magnetic_x = df['magnetic_x']
    magnetic_y = df['magnetic_y']
    magnetic_z = df['magnetic_z']
    step = df['step']
    # find split step line number
    # print(step[step == 1].index[-1])
    for i in range(len(grid)):
        line_split.append((step[step == i].index[0], step[step == i].index[-1]))
    for i in range(len(line_split)):
        file_line_write.extend(grid[i])
        # file_line_write.append( magnetic_x[line_split[i][0]:line_split[i][1]].mean() )
        # file_line_write.append( magnetic_y[line_split[i][0]:line_split[i][1]].mean() )
        # file_line_write.append( magnetic_z[line_split[i][0]:line_split[i][1]].mean() )
        file_line_write.append( magnetic_x[(line_split[i][0]+line_split[i][1])/2] )
        file_line_write.append( magnetic_y[(line_split[i][0]+line_split[i][1])/2] )
        file_line_write.append( magnetic_z[(line_split[i][0]+line_split[i][1])/2] )
        file_line_write.append( orientation[line_split[i][0]:line_split[i][1]].mean() )
        file_line_write.append(0)
        file_line_write.append(0)
        # f.write(str)
        file_write.append(file_line_write)
        file_line_write = []

    save = pd.DataFrame(file_write)
    save.to_csv(file_path+'_'+file_name, index=False, header = False, sep=',')
    print("done")

if __name__ == '__main__':

    # load_step(file_name = 'step_7_0__7_12.csv')
    # load_step(file_path = 'new_trace_data/', file_name = '8.4-8.12-15.12.csv', grid = [[8, 4],[8,5],[8,6], [8,7],[8,8],[8,9],[8,10],[8,11],[8,12],[9,12],[10,12],[11,12],[12,12],[13,12],[14,12],[15,12]])
    # load_step(file_path = 'new_trace_data/', file_name = '5.8-5.16-15.16.csv', grid =  [[5,8],[5,9],[5,10],[5,11],[5,12],[5,13],[5,14],[5,15],[5,16],[6,16],[7,16],[8,16],[9,16],[10,16],[11,16],[12,16],[13,16],[14,16],[15,16]])
    load_step(file_path = 'new_trace_data/', file_name = '5.18-5.25-15.25.csv', grid = [[5,18],[5,19],[5,20],[5,21],[5,22],[5,23],[5,24],[5,25],[6,25],[7,25],[8,25],[9,25],[10,25],[11,25],[12,25],[13,25],[14,25],[15,25]])
    '''
    3 hyper parameter: step model, offset of orientation, threshold of megnatic value
    1. add grid list
    2. change load_step() parameter
    3. run: python read_step_all.py XX, XX is offset of orientation error(by device)
    4. chenge parameter in localiaztion.get_step_raw_data()
    5. fill real_step = [] in main segment, run python localization.py
    6. get accurancy, copy print result: mean_path and do plot
    7. copy accurancy to plot_cdf.py
    '''


