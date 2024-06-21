import numpy as np
import matplotlib.pyplot as plt
import akshare as ak


# 获取股票数据
def get_stock_data(symbol, start_date, end_date):
    stock_data = ak.stock_zh_index_daily_em(symbol=symbol, start_date=start_date, end_date=end_date)
    # stock_data['date'] = stock_data['date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
    return stock_data


# 数据离散化
def discretize_data(data, num_states):
    min_change = -0.05
    max_change = 0.05
    interval = (max_change - min_change) / (num_states - 1)
    bins = np.linspace(min_change, max_change, num_states)

    price_changes = np.diff(data['close']) / data['close'][:-1]
    discretized_data = np.digitize(price_changes, bins=bins)

    return discretized_data


# 频率统计
def calculate_transition_probabilities(discretized_data, num_states):
    state_counts = np.zeros((num_states+1, num_states+1))
    for i in range(len(discretized_data) - 1):
        state_counts[discretized_data[i], discretized_data[i + 1]] += 1
    transition_probabilities = state_counts / np.sum(state_counts, axis=1, keepdims=True)
    return transition_probabilities


# 作出转移矩阵
def create_transition_matrix(transition_probabilities):
    # 创建转移矩阵

    return transition_matrix


# 转移矩阵实验
def experiment_transition_matrix(transition_matrix, max_iterations=1000, epsilon=1e-6):
    prev_transition_matrix = np.copy(transition_matrix)
    for i in range(max_iterations):
        transition_matrix = np.dot(transition_matrix, transition_matrix)
        if np.linalg.norm(transition_matrix - prev_transition_matrix) < epsilon:
            convergence_time = i
            break
        prev_transition_matrix = np.copy(transition_matrix)
    else:
        convergence_time = max_iterations
    return convergence_time, transition_matrix


# 转移矩阵可视化
def visualize_transition_matrix(transition_matrix):
    plt.figure(figsize=(10, 6))
    num_states = transition_matrix.shape[0]
    for current_state in range(num_states):
        plt.bar(range(num_states), transition_matrix[current_state, :])
    plt.xlabel('Next State')
    plt.ylabel('Transition Probability')
    plt.title('Transition Matrix (Curve Chart)')
    plt.legend()
    plt.grid(True)
    plt.show()



# 主函数
def main():
    # 获取股票数据
    # 获取训练集和验证集数据
    train_data = get_stock_data(symbol="sz399552", start_date="20000101", end_date="20190101")
    validation_data = get_stock_data(symbol="sz399552", start_date="20190101", end_date="20200101")
    print("训练集数据量:", len(train_data))
    print("验证集数据量:", len(validation_data))

    # 数据离散化
    # 定义离散化状态数量
    num_states = 30
    # 对训练集和验证集进行离散化
    train_discretized_data = discretize_data(train_data, num_states)
    validation_discretized_data = discretize_data(validation_data, num_states)
    print("训练集离散化后数据:", train_discretized_data)
    print("验证集离散化后数据:", validation_discretized_data)

    # 频率统计
    # 计算训练集和验证集的转移概率
    train_transition_probabilities = calculate_transition_probabilities(train_discretized_data, num_states)
    validation_transition_probabilities = calculate_transition_probabilities(validation_discretized_data, num_states)
    print("训练集转移概率矩阵:")
    print(train_transition_probabilities)
    print("验证集转移概率矩阵:")
    print(validation_transition_probabilities)

    # 创建转移矩阵
    # transition_matrix = create_transition_matrix(transition_probabilities)

    # 转移矩阵实验
    # 训练集转移矩阵实验
    train_convergence_time, train_final_transition_matrix = experiment_transition_matrix(train_transition_probabilities)
    print("训练集转移矩阵实验收敛时间:", train_convergence_time)
    # 验证集转移矩阵实验
    validation_convergence_time, validation_final_transition_matrix = experiment_transition_matrix(
        validation_transition_probabilities)
    print("验证集转移矩阵实验收敛时间:", validation_convergence_time)

    # 转移矩阵可视化
    visualize_transition_matrix(train_final_transition_matrix)
    visualize_transition_matrix(validation_final_transition_matrix)


if __name__ == "__main__":
    main()