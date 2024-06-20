import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 数据获取和准备
symbol = "sz000002"  # 万科A
data_train = ak.stock_zh_a_daily(symbol=symbol, start_date="20000101", end_date="20190101")
data_test = ak.stock_zh_a_daily(symbol=symbol, start_date="20190101", end_date="20191231")

# 计算百分比日变化
data_train['pct_change'] = data_train['close'].pct_change() * 100

# 数据离散化
N = 10  # 状态数量
bins = np.linspace(-5, 5, N+1)  # 创建等距bins
labels = range(N)
data_train['state'] = pd.cut(data_train['pct_change'], bins=bins, labels=labels, include_lowest=True)

# 频率统计与转移矩阵
transition_matrix = np.zeros((N, N))

for i in range(1, len(data_train)):
    current_state = data_train.iloc[i-1]['state']
    next_state = data_train.iloc[i]['state']
    if pd.notna(current_state) and pd.notna(next_state):
        transition_matrix[int(current_state), int(next_state)] += 1

# 标准化转移矩阵
transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)

# 迭代检查收敛性
iterations = 100
tm_pow = transition_matrix.copy()
for _ in range(iterations):
    tm_pow = np.dot(tm_pow, transition_matrix)

# 绘制稳态转移矩阵
plt.figure(figsize=(8, 6))
plt.imshow(tm_pow, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title("Steady State Transition Matrix After 100 Iterations")
plt.xlabel("Next State")
plt.ylabel("Current State")
plt.show()

# 统计每个状态的天数
state_counts = data_train['state'].value_counts().sort_index()

# 将状态标签改为区间标签
state_labels = [f"{(bins[i]+bins[i+1])/2}%" for i in range(N)]

# 绘制折线图
plt.figure(figsize=(10, 6))
plt.plot(state_labels, state_counts.values, marker='o', linestyle='-')
plt.title('Days Count for Each State (-5% to 5%)')
plt.xlabel('State')
plt.ylabel('Days Count')
plt.xticks(rotation=45)  # 旋转x轴刻度标签以免重叠
plt.grid(True)
plt.show()


# 提取股票价格数据
stock_prices = data_train['close']

# 绘制股票价格变化曲线
plt.figure(figsize=(10, 6))
plt.plot(stock_prices, color='blue')
plt.title('Stock Price Change Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.show()


# 对比变化：变更离散策略等
# （可按需实现）