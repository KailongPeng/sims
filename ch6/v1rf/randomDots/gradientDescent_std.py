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
n = 20
# 定义控制点随机移动距离的缩放因子
lambda_factor = 0.01  # 示例缩放因子
# 定义优化迭代的次数
iterations = 50  # 优化迭代次数

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
    m1_distanceMatrix = initial_dist_matrix[np.triu_indices(n, k=1)]
    m2_distanceMatrix = new_dist_matrix[np.triu_indices(n, k=1)]

    # 将移动前的元素取相反数作为x轴，移动后的元素与移动前的元素差作为y轴
    x_coactivation = - m1_distanceMatrix  # m 越大, 越differentiation, 越少的co-activation
    y_integration = - (m2_distanceMatrix - m1_distanceMatrix)  # m 越大, 越differentiation, 因此

    # 根据x轴的值进行排序
    sorted_indices = np.argsort(x_coactivation)
    x_sorted = x_coactivation[sorted_indices]
    y_sorted = y_integration[sorted_indices]

    # 将排序后的x轴值分为三部分：noChange, differentiation, integration
    third = len(x_sorted) // 3
    noChange = y_sorted[:third]
    differentiation = y_sorted[third:2 * third]
    integration = y_sorted[2 * third:]

    # 计算三部分的目标函数值
    mean_noChange, std_noChange = np.mean(noChange), np.std(noChange)
    mean_differentiation, std_differentiation = np.mean(differentiation), np.std(differentiation)
    mean_integration, std_integration = np.mean(integration), np.std(integration)

    obj_noChange = mean_noChange**2 + std_noChange  # mean_noChange越接近0，std_noChange越小，目标函数值越小
    obj_differentiation = mean_differentiation + std_differentiation  # mean_differentiation越小于0，std_differentiation越小，目标函数值越小
    obj_integration = - mean_integration + std_integration  # mean_integration越大于0，std_integration越小，目标函数值越小

    # 计算总体目标函数值
    objective = obj_noChange + obj_differentiation + obj_integration
    return objective, x_coactivation, y_integration, mean_noChange, std_noChange, mean_differentiation, std_differentiation, mean_integration, std_integration


def calculate_gradient(points, initial_dist_matrix):
    gradient = np.zeros_like(points)
    h = 1e-5  # 微小的扰动，用于计算数值梯度
    for i in range(n):
        for j in range(2):  # 计算每个点的x和y方向上的梯度
            points[i, j] += h
            dist_matrix_plus_h = calculate_distance_matrix(points)
            (_, _, _,
             mean_noChange_plus_h, std_noChange_plus_h,
             mean_differentiation_plus_h, std_differentiation_plus_h,
             mean_integration_plus_h, std_integration_plus_h) = calculate_objective_and_plot_data(initial_dist_matrix, dist_matrix_plus_h)

            points[i, j] -= 2 * h
            dist_matrix_minus_h = calculate_distance_matrix(points)
            (_, _, _,
             mean_noChange_minus_h, std_noChange_minus_h,
             mean_differentiation_minus_h, std_differentiation_minus_h,
             mean_integration_minus_h, std_integration_minus_h) = calculate_objective_and_plot_data(initial_dist_matrix, dist_matrix_minus_h)

            points[i, j] += h  # 恢复点的位置

            # 计算每个目标函数的梯度
            # mean_noChange更接近0，std_noChange更大，mean_differentiation更小，std_differentiation更小， mean_integration 更大，std_integration更小
            # 总结来说： mean_noChange更接近0，mean_integration更大
            grad_mean_noChange = (mean_noChange_plus_h**2 - mean_noChange_minus_h**2) / (2 * h)  # 注意这里是 mean_noChange 的平方， 因为我们希望它更接近0
            grad_std_noChange = (std_noChange_plus_h - std_noChange_minus_h) / (2 * h)
            grad_mean_differentiation = (mean_differentiation_plus_h - mean_differentiation_minus_h) / (2 * h)
            grad_std_differentiation = (std_differentiation_plus_h - std_differentiation_minus_h) / (2 * h)
            grad_mean_integration = (mean_integration_plus_h - mean_integration_minus_h) / (2 * h)
            grad_std_integration = (std_integration_plus_h - std_integration_minus_h) / (2 * h)

            # 梯度归一化
            grad_mean_noChange /= np.linalg.norm(grad_mean_noChange) + 1e-8  # 防止除以零
            grad_std_noChange /= np.linalg.norm(grad_std_noChange) + 1e-8
            grad_mean_differentiation /= np.linalg.norm(grad_mean_differentiation) + 1e-8
            grad_std_differentiation /= np.linalg.norm(grad_std_differentiation) + 1e-8
            grad_mean_integration /= np.linalg.norm(grad_mean_integration) + 1e-8
            grad_std_integration /= np.linalg.norm(grad_std_integration) + 1e-8

            # 设置权重
            # weights = {
            #     'mean_noChange': 1,
            #     'std_noChange': 1,
            #     'mean_differentiation': 1,
            #     'std_differentiation': 1,
            #     'mean_integration': 1,
            #     'std_integration': 1
            # }
            weights = {
                'mean_noChange': 0,
                'std_noChange': 0,
                'mean_differentiation': 1,
                'std_differentiation': 0,
                'mean_integration': 0,
                'std_integration': 0
            }

            # 加和归一化后的梯度
            gradient[i, j] = (
                    grad_mean_noChange * weights['mean_noChange']
                    + grad_std_noChange * weights['std_noChange']
                    + grad_mean_differentiation * weights['mean_differentiation']
                    + grad_std_differentiation * weights['std_differentiation']
                    - grad_mean_integration * weights['mean_integration']  # 注意 mean_integration 的符号为负，因为我们希望它更大
                    + grad_std_integration * weights['std_integration']
            )

    return gradient

def move_points_randomly(points, lambda_factor):
    # 随机移动点，移动后的点位置是原始位置加上一个随机值乘以lambda_factor
    new_points = points + (np.random.rand(n, 2) - 0.5) * lambda_factor
    return new_points

# 计算初始目标函数值和散点图数据
new_points = move_points_randomly(points, lambda_factor)  # 随机移动点
new_dist_matrix = calculate_distance_matrix(new_points)  # 计算新的距离矩阵
(initial_objective, initial_x, initial_y,
 initial_mean_noChange, initial_std_noChange,
 initial_mean_differentiation, initial_std_differentiation,
 initial_mean_integration, initial_std_integration) = calculate_objective_and_plot_data(initial_dist_matrix, new_dist_matrix)

# 优化点的位置
best_points = points.copy()  # 复制初始点集作为最佳点集的初始值
best_objective = initial_objective  # 初始化最佳目标函数值为初始值
best_mean_noChange = initial_mean_noChange
best_std_noChange = initial_std_noChange
best_mean_differentiation = initial_mean_differentiation
best_std_differentiation = initial_std_differentiation
best_mean_integration = initial_mean_integration
best_std_integration = initial_std_integration
best_x = initial_x
best_y = initial_y

# 存储每次迭代的目标函数值、t值和p值以绘制损失曲线
objectives = [initial_objective]
mean_noChange_list = [initial_mean_noChange]
std_noChange_list = [initial_std_noChange]
mean_differentiation_list = [initial_mean_differentiation]
std_differentiation_list = [initial_std_differentiation]
mean_integration_list = [initial_mean_integration]
std_integration_list = [initial_std_integration]

# 进行迭代优化
for _ in tqdm(range(iterations)):
    gradient = calculate_gradient(best_points, initial_dist_matrix)  # 计算梯度
    # new_points = best_points - lambda_factor * gradient  # 沿负梯度方向更新点的位置
    learningRate = 1e-8 #0.000001
    # new_points = best_points - gradient / np.max(np.abs(gradient)) * learningRate  # 沿负梯度方向更新点的位置
    new_points = best_points - gradient * learningRate  # 沿负梯度方向更新点的位置
    new_dist_matrix = calculate_distance_matrix(new_points)  # 计算新的距离矩阵
    (new_objective, new_x, new_y,
     mean_noChange, std_noChange,
     mean_differentiation, std_differentiation,
     mean_integration, std_integration) = calculate_objective_and_plot_data(initial_dist_matrix, new_dist_matrix)  # 计算新的目标函数值和散点图数据

    best_points = new_points.copy()
    best_objective = new_objective
    best_mean_noChange = mean_noChange
    best_std_noChange = std_noChange
    best_mean_differentiation = mean_differentiation
    best_std_differentiation = std_differentiation
    best_mean_integration = mean_integration
    best_std_integration = std_integration
    best_x = new_x
    best_y = new_y

    objectives.append(new_objective)
    mean_noChange_list.append(mean_noChange)
    std_noChange_list.append(std_noChange)
    mean_differentiation_list.append(mean_differentiation)
    std_differentiation_list.append(std_differentiation)
    mean_integration_list.append(mean_integration)
    std_integration_list.append(std_integration)


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
# 画一条 y=0 的参考线
plt.axhline(y=0, color='gray', linestyle='--')

plt.subplot(2, 2, 4)
plt.scatter(best_x, best_y, c='red', label='最终')
plt.title("最终目标函数散点图")
plt.xlabel("-m1")
plt.ylabel("m2 - m1")
plt.legend()
plt.axhline(y=0, color='gray', linestyle='--')

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


# 绘制t值和p值曲线
plt.figure(figsize=(15, 10))

plt.subplot(3, 2, 1)
plt.plot(mean_noChange_list, label='mean_noChange')
plt.title("mean_noChange 曲线, 应该接近0")  # mean_noChange更接近0，mean_integration更大
plt.xlabel("迭代次数")
plt.ylabel("mean_noChange")
plt.legend()
plt.axhline(y=0, color='gray', linestyle='--')

plt.subplot(3, 2, 2)
plt.plot(std_noChange_list, label='std_noChange')
plt.title("std_noChange 曲线")
plt.xlabel("迭代次数")
plt.ylabel("std_noChange")
plt.legend()
plt.axhline(y=0, color='gray', linestyle='--')

plt.subplot(3, 2, 3)
plt.plot(mean_differentiation_list, label='mean_differentiation')
plt.title("mean_differentiation 曲线")
plt.xlabel("迭代次数")
plt.ylabel("mean_differentiation")
plt.legend()
plt.axhline(y=0, color='gray', linestyle='--')

plt.subplot(3, 2, 4)
plt.plot(std_differentiation_list, label='std_differentiation')
plt.title("std_differentiation 曲线")
plt.xlabel("迭代次数")
plt.ylabel("std_differentiation")
plt.legend()
plt.axhline(y=0, color='gray', linestyle='--')

plt.subplot(3, 2, 5)
plt.plot(mean_integration_list, label='mean_integration')
plt.title("mean_integration 曲线， 应该更大")  # mean_noChange更接近0，mean_integration更大
plt.xlabel("迭代次数")
plt.ylabel("mean_integration")
plt.legend()
plt.axhline(y=0, color='gray', linestyle='--')

plt.subplot(3, 2, 6)
plt.plot(std_integration_list, label='std_integration')
plt.title("std_integration 曲线")
plt.xlabel("迭代次数")
plt.ylabel("std_integration")
plt.legend()
plt.axhline(y=0, color='gray', linestyle='--')

plt.tight_layout()
plt.show()
