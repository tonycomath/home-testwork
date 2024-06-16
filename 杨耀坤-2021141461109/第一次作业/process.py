import akshare
import os

import numpy as np
import pandas as pd

import draw


''' 
function: 离散化连续的股价
'''
class Partition:
    '''
    function: 构造函数
    params:
        min: 股价的最小值
        max: 股价的最大值
        number: 划分的区间
    '''
    def __init__(self, min, max, number):
        self.__bound = []
        self.__center = []
        delta = (max - min) / number
        for i in range(number):
            self.__bound.append(min + (i + 1) * delta)
            self.__center.append(min + (i + 0.5) * delta)
    
    '''
    function: 返回指定下标对应的点的值
    params:
        index: 指定的下标 
    '''
    def __indexValue(self, index):
        return self.__center[index]
    
    '''
    function: 返回第一个大于特定数的中心的下标
    params:
        number: 特定的数
    '''
    def indexOf(self, number):
        l = 0 
        r = len(self.__center)
        ans = 0
        while l <= r:
            mid = (l + r) >> 1
            if self.__bound[mid] >= number:
                ans = mid
                r = mid - 1
            else:
                l = mid + 1
        return ans

    '''
    function: 返回将连续的股价进行离散化后的值
    params:
        number: 股价
    ''' 
    def map(self, number):
        return self.__indexValue([self.indexOf(number)])

''' 
function: 马尔科夫矩阵
'''
class Matrix:
    ''' 
    function: 构造函数
    params:
        size: 矩阵的大小
    '''
    def __init__(self, size):
        self.matrix = [[0 for _ in range(size)] for _ in range(size)]

    ''' 
    function: 修改矩阵
    params:
        x: 矩阵的第一维坐标
        y: 矩阵的第二维坐标
    '''
    def append(self, x, y): # matrix(x, y): x -> y
        self.matrix[x][y] += 1
    
    ''' 
    function: 将矩阵归一化
    '''
    def normalize(self):
        for i in range(len(self.matrix)):
            self.matrix[i] = [x/sum(self.matrix[i]) for x in self.matrix[i]]
        return np.array(self.matrix)

    ''' 
    function: 判断矩阵的敛散性
    params:
        depth: 最大迭代次数
        delta_tp: 
    '''
    def isConvergence(self, depth = 10000, delta_tp = 1e-8):

        p = np.array(self.matrix, dtype = np.float64)
        tp = np.array(self.matrix, dtype = np.float64)

        std_tp = np.std(tp) # 计算方差
        
        step = 1
        while step <= depth:
            tp = np.dot(tp, p)
            step += 1
            std_tp_new = np.std(tp)

            if abs(std_tp - std_tp_new) < delta_tp:
                return True, step, tp
            std_tp = std_tp_new

        return False, step, tp



if __name__ == '__main__':
    # 参数
    PARTITION = 10

    # 预处理
    filePath = "data.csv"
    if os.path.exists(filePath):
        dataSrc = pd.read_csv(filePath)
    else:
        dataSrc = akshare.stock_zh_index_daily_em("sz000001", "20000101", "20191231")
        dataSrc.to_csv(filePath)


    data = dataSrc.loc[:, ['date', 'close']]
    data.set_index('date', inplace = True) 
    data.rename(columns = {'close': 'value'}, inplace = True)
    partition = Partition(data.min().values[0], data.max().values[0], PARTITION)
    data['index'] = data.apply(lambda row: partition.indexOf(row['value']), axis = 1)


    # 数据集划分
    train_set = data.loc['2000-01-04':'2018-12-31', :]
    valid_set = data.loc['2019-01-01':'2019-12-31', :]

    # 模型
    m = Matrix(PARTITION)

    for i in range(len(train_set['index']) - 1):
        m.append(train_set['index'].iloc[i], train_set['index'].iloc[i+1])

    p = m.normalize()
    result, step, tp = m.isConvergence()

    start_index =  valid_set['index'].iloc[0]
    start_price = [0 for _ in range(start_index)]
    start_price.append(1)
    start_price = start_price + [0 for _ in range(9 - start_index)] 
    start_price = np.array(start_price)

    y_predict = []
    y_real_noramlized = []
    y_predict.append(partition.indexValue(start_index))
    y_real_noramlized.append(partition.indexValue(start_index))
    

    for i in range(len(valid_set['value']) - 1):
        start_price = np.dot(start_price, p)
        y_predict.append(partition.indexValue(np.argmax(start_price)))
        y_real_noramlized.append(partition.indexValue(valid_set['index'].iloc[i+1]))


    
    if result:
        center = [round(num, 3) for num in partition.center]
        draw.heatmap(x = center, y = center, value = p.round(3), title = "Transition matrix (before convergence)", save_name = 'tmbc')
        draw.heatmap(x = center, y = center, value = tp.round(3), title = "Transition matrix (after convergence)", save_name = 'tmac')
    else:
        print('Fail to converge')

    draw.lineBarChart(y_predict = y_predict, y_real = valid_set['value'], y_real_normalized = y_real_noramlized, save_name = 'comparison')






