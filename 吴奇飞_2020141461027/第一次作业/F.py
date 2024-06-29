import math
import akshare
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# 检查转移矩阵是否收敛
def check_convergence(transition_matrix, max_iterations=1000, tolerance=1e-6):
    current_matrix = transition_matrix
    matrix_step = [current_matrix]  # 存储每一步的矩阵

    for i in range(max_iterations):
        next_matrix = np.dot(current_matrix, transition_matrix)  # 计算下一步的矩阵
        matrix_step.append(next_matrix)

        # 检查当前矩阵和下一步矩阵是否在容差范围内接近
        if np.allclose(current_matrix, next_matrix, atol=tolerance):
            return True, matrix_step  # 如果收敛，返回True和矩阵序列
        current_matrix = next_matrix

    return False, matrix_step  # 如果未收敛，返回False和矩阵序列


# 可视化数据变化
def visualize_data_changes(data):
    data.plot(marker='o', linestyle='')  # 绘制散点图
    plt.title('Scatter Plot of DataFrame')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.show()


# 可视化转移矩阵的变化
def visualize_transition_matrix_changes(matrix_list, nums):
    row_ = int(nums / 5)
    col_ = 5
    fig, axs = plt.subplots(nrows=row_, ncols=col_)

    len1 = 2 * math.sqrt(len(matrix_list))
    delta1 = len1 * 2 / nums

    # 绘制前一部分的矩阵变化
    for i in range(nums // 2):
        row = i // col_
        col = i % col_
        axs[row, col].imshow(matrix_list[int(i * delta1)], cmap='hot', interpolation='nearest')
        axs[row, col].set_title("iter%s" % int(i * delta1))

    len2 = len(matrix_list) - len1
    delta2 = len2 * 2 / nums

    # 绘制后一部分的矩阵变化
    for i in range(nums // 2, nums - 1):
        row = i // col_
        col = i % col_
        axs[row, col].imshow(matrix_list[int(len1 + delta2 * (i - nums // 2))], cmap='hot', interpolation='nearest')
        axs[row, col].set_title("iter%s" % int(len1 + delta2 * (i - nums // 2)))

    # 绘制最后一个矩阵
    axs[row_ - 1, col_ - 1].imshow(matrix_list[-1], cmap='hot', interpolation='nearest')
    axs[row_ - 1, col_ - 1].set_title("iter%s" % (len(matrix_list) - 1))
    plt.tight_layout()
    plt.show()


# 主函数
def main():
    # 获取训练数据和测试数据
    data_train = akshare.stock_zh_index_daily_em(symbol="csi931151", start_date="19900101", end_date="20240101")
    data_test = akshare.stock_zh_index_daily_em(symbol="csi931151", start_date="20240101", end_date="20500101")

    N = 10  # 定义状态数量
    # 将数据分箱并添加标签
    data_train['open_label'] = pd.cut(data_train['open'], bins=N, labels=False, right=False)
    bin_edges = pd.cut(data_train['open'], bins=N, retbins=True, right=False)[1]
    data_train['open_1'] = pd.cut(data_train['open'], bins=N, labels=bin_edges[0:-1], right=False)
    data = data_train['open_label']

    # 初始化转移计数矩阵
    transition_counts = np.zeros((N, N))
    for i in range(len(data) - 1):
        current_state = data[i]
        next_state = data[i + 1]
        transition_counts[current_state][next_state] += 1

    # 计算转移概率矩阵
    transition_array = transition_counts / transition_counts.sum(axis=1, keepdims=True)

    # 检查转移矩阵是否收敛
    convergence, array_list = check_convergence(transition_array)
    if convergence:
        print("转移矩阵迭代%s次后收敛" % (len(array_list) - 1))
        print("收敛矩阵为:",)
        print(array_list[-1])
    else:
        print("转移矩阵不收敛")

    # 可视化转移矩阵的变化
    visualize_transition_matrix_changes(array_list, 20)
    # 可视化数据变化
    visualize_data_changes(data_train['open'])


# 执行主函数
main()
