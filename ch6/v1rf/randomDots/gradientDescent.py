import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy.stats import ttest_1samp
from tqdm import tqdm

# 设置 matplotlib 的默认字体为黑体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

import numpy as np

# 设置随机种子
np.random.seed(42)  # 42是种子值，你可以选择任何整数作为种子值

# 定义点的数量
n = 21
# 定义控制点随机移动距离的缩放因子
lambda_factor = 0.01  # 示例缩放因子
# 定义优化迭代的次数
iterations = 1000  # 优化迭代次数

# 在0到1的二维平面上随机均匀分布n个点
points = np.random.rand(n, 2)

# 定义一个函数，用于计算点集的距离矩阵
def calculate_distance_matrix(points):
    dist_matrix = np.zeros((n, n))  # 初始化距离矩阵为零
    for i in range(n):
        for j in range(n):
            if i != j:  # 排除对角线上的元素
                dist_matrix[i, j] = np.linalg.norm(points[i] - points[j])  # 计算两点间的距离
    return dist_matrix

# 计算初始点集的距离矩阵
initial_dist_matrix = calculate_distance_matrix(points)

# 定义一个函数，用于计算目标函数和散点图数据
def calculate_objective_and_plot_data(initial_dist_matrix, new_dist_matrix):
    # 计算移动前后的距离矩阵的上三角元素
    m1 = initial_dist_matrix[np.triu_indices(n, k=1)]
    m2 = new_dist_matrix[np.triu_indices(n, k=1)]

    # 将移动前的元素取相反数作为x轴，移动后的元素与移动前的元素差作为y轴
    x = -m1
    y = m2 - m1

    # 根据x轴的值进行排序
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]

    # 将排序后的x轴值分为三部分：noChange, differentiation, integration
    third = len(x_sorted) // 3
    noChange = y_sorted[:third]
    differentiation = y_sorted[third:2 * third]
    integration = y_sorted[2 * third:]

    # 计算三部分的目标函数值
    t_noChange, p_noChange = ttest_1samp(noChange, 0)
    t_differentiation, p_differentiation = ttest_1samp(differentiation, 0)
    t_integration, p_integration = ttest_1samp(integration, 0)

    obj_noChange = 1000 * np.abs(t_noChange) + p_noChange
    obj_differentiation = t_differentiation + p_differentiation
    obj_integration = -t_integration + p_integration

    # 计算总体目标函数值
    objective = obj_noChange + obj_differentiation + obj_integration
    return objective, x, y, t_noChange, p_noChange, t_differentiation, p_differentiation, t_integration, p_integration


# 定义一个函数，用于计算梯度
def calculate_gradient(points, initial_dist_matrix):
    gradient = np.zeros_like(points)
    h = 1e-5  # 微小的扰动，用于计算数值梯度
    for i in range(n):
        for j in range(2):  # 计算每个点的x和y方向上的梯度
            points[i, j] += h
            dist_matrix_plus_h = calculate_distance_matrix(points)
            objective_plus_h, _, _, _, _, _, _, _, _ = calculate_objective_and_plot_data(initial_dist_matrix, dist_matrix_plus_h)

            points[i, j] -= 2 * h
            dist_matrix_minus_h = calculate_distance_matrix(points)
            objective_minus_h, _, _, _, _, _, _, _, _ = calculate_objective_and_plot_data(initial_dist_matrix, dist_matrix_minus_h)

            points[i, j] += h  # 恢复点的位置
            gradient[i, j] = (objective_plus_h - objective_minus_h) / (2 * h)
    return gradient

# 计算初始目标函数值和散点图数据
new_points = points.copy()  # 初始点集作为最佳点集的初始值
new_dist_matrix = calculate_distance_matrix(new_points)  # 计算新的距离矩阵
(initial_objective, initial_x, initial_y,
 initial_t_noChange, initial_p_noChange,
 initial_t_differentiation, initial_p_differentiation,
 initial_t_integration, initial_p_integration) = calculate_objective_and_plot_data(initial_dist_matrix, new_dist_matrix)

# 优化点的位置
best_points = points.copy()  # 复制初始点集作为最佳点集的初始值
best_objective = initial_objective  # 初始化最佳目标函数值为初始值

# 存储每次迭代的目标函数值以绘制损失曲线
objectives = [initial_objective]

# 进行迭代优化
for _ in tqdm(range(iterations)):
    gradient = calculate_gradient(best_points, initial_dist_matrix)  # 计算梯度
    new_points = best_points - lambda_factor * gradient  # 沿负梯度方向更新点的位置
    new_dist_matrix = calculate_distance_matrix(new_points)  # 计算新的距离矩阵
    new_objective, new_x, new_y, t_noChange, p_noChange, t_differentiation, p_differentiation, t_integration, p_integration = calculate_objective_and_plot_data(initial_dist_matrix, new_dist_matrix)  # 计算新的目标函数值和散点图数据

    # 如果新的目标函数值小于最佳目标函数值，则更新最佳点集和最佳目标函数值
    if new_objective < best_objective:
        best_points = new_points.copy()
        best_objective = new_objective
        best_x = new_x
        best_y = new_y

    objectives.append(new_objective)

# 绘制初始和最终的点集位置
plt.figure(figsize=(12, 12))

plt.subplot(2, 2, 1)
plt.scatter(points[:, 0], points[:, 1], c='blue', label='初始')
plt.title("初始位置")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()

plt.subplot(2, 2, 2)
plt.scatter(best_points[:, 0], best_points[:, 1], c='red', label='最终')
plt.title("最终位置")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()

# 设置相同的xlim和ylim
xlim = (min(points[:, 0].min(), best_points[:, 0].min())-0.1, max(points[:, 0].max(), best_points[:, 0].max())+0.1)
ylim = (min(points[:, 1].min(), best_points[:, 1].min())-0.1, max(points[:, 1].max(), best_points[:, 1].max())+0.1)

plt.subplot(2, 2, 1)
plt.xlim(xlim)
plt.ylim(ylim)

plt.subplot(2, 2, 2)
plt.xlim(xlim)
plt.ylim(ylim)

# 绘制初始和最终的目标函数散点图
plt.subplot(2, 2, 3)
plt.scatter(initial_x, initial_y, c='blue', label='初始')
plt.title("初始目标函数散点图")
plt.xlabel("-m1")
plt.ylabel("m2 - m1")
plt.legend()

plt.subplot(2, 2, 4)
plt.scatter(best_x, best_y, c='red', label='最终')
plt.title("最终目标函数散点图")
plt.xlabel("-m1")
plt.ylabel("m2 - m1")
plt.legend()

# 调整子图布局并显示所有子图
plt.tight_layout()
plt.show()

# 绘制损失曲线
plt.figure()
plt.plot(objectives, label='目标函数值')
plt.title("损失曲线")
plt.xlabel("迭代次数")
plt.ylabel("目标函数值")
plt.legend()
plt.show()
