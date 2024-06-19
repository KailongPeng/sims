"""
假代码

对于n=21个点，首先把这些点随机均匀平铺在一个0到1的二维平面上，然后把这21个点随机向任意方向移动一点点，这个随机移动的距离被scalling factor lambda控制。

计算objective：
对于输入的n个点，计算出移动前和移动后分别的两个距离矩阵 n x n 。这两个距离矩阵中的上半矩阵元素数量为n*(n-1)/2个。移动前的n*(n-1)/2个元素记为m1，移动后的n*(n-1)/2个元素记为m2.
将m1值的相反数作为x轴，将m2-m1的值作为y轴，画一个散点图。
对于这个散点图，获取x轴上面的最小的n*(n-1)/2/3个元素记为 noChange, 获取x轴上面的最大的n*(n-1)/2/3个元素记为 integration, 剩下的n*(n-1)/2/3个元素记为differentiation.
对于noChange,计算这n*(n-1)/2/3个元素的y轴的与0相比的t test 的 t 值的绝对值加上 t test的p value, 记为 obj_noChange.
对于differentiation,计算这n*(n-1)/2/3个元素的y轴的与0相比的t值加上 t test的p value, 记为 obj_differentiation.
对于integration,计算这n*(n-1)/2/3个元素的y轴的与0相比的t值的相反数加上 t test的p value, 记为 obj_integration.

总体的objective = obj_noChange + obj_differentiation + obj_integration.

现在对于这21个点, 轮流对他们进行随机的移动, 这个随机移动的距离被scalling factor lambda控制。每一个移动之后, 计算一次objective, 如果objective变小,则保留这次移动, 如果objective不变或者变大,则放弃这次移动.

最后, 展示最开始没有移动的21个点在二维平面上的位置,以及最终完成所有的移动之后的21个点在二维平面上的位置.

与此同时,也要展示最开始和结束的时候的objective的x轴和y轴的散点图

还要记录然后最终画出训练过程的loss curve

"""

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
iterations = 10  # 优化迭代次数

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

    obj_noChange = 10 * t_noChange**2 + p_noChange  # t_noChange越接近0，p_noChange越小，目标函数值越小
    obj_differentiation = t_differentiation + p_differentiation  # t_differentiation越小于0，p_differentiation越小，目标函数值越小
    obj_integration = -t_integration + p_integration  # t_integration越大于0，p_integration越小，目标函数值越小

    # 计算总体目标函数值
    objective = obj_noChange + obj_differentiation + obj_integration
    return objective, x, y, t_noChange, p_noChange, t_differentiation, p_differentiation, t_integration, p_integration


# # 定义一个函数，用于计算梯度
# def calculate_gradient(points, initial_dist_matrix):
#     gradient = np.zeros_like(points)
#     h = 1e-5  # 微小的扰动，用于计算数值梯度
#     for i in range(n):
#         for j in range(2):  # 计算每个点的x和y方向上的梯度
#             points[i, j] += h
#             dist_matrix_plus_h = calculate_distance_matrix(points)
#             objective_plus_h, _, _, _, _, _, _, _, _ = calculate_objective_and_plot_data(initial_dist_matrix, dist_matrix_plus_h)
#
#             points[i, j] -= 2 * h
#             dist_matrix_minus_h = calculate_distance_matrix(points)
#             objective_minus_h, _, _, _, _, _, _, _, _ = calculate_objective_and_plot_data(initial_dist_matrix, dist_matrix_minus_h)
#
#             points[i, j] += h  # 恢复点的位置
#             gradient[i, j] = (objective_plus_h - objective_minus_h) / (2 * h)
#     return gradient
# 定义一个函数，用于计算各个目标函数的梯度
def calculate_gradient(points, initial_dist_matrix):
    gradient = np.zeros_like(points)
    h = 1e-5  # 微小的扰动，用于计算数值梯度
    for i in range(n):
        for j in range(2):  # 计算每个点的x和y方向上的梯度
            points[i, j] += h
            dist_matrix_plus_h = calculate_distance_matrix(points)
            (_, _, _,
             t_noChange_plus_h, p_noChange_plus_h,
             t_differentiation_plus_h, p_differentiation_plus_h,
             t_integration_plus_h, p_integration_plus_h) = calculate_objective_and_plot_data(initial_dist_matrix, dist_matrix_plus_h)

            points[i, j] -= 2 * h
            dist_matrix_minus_h = calculate_distance_matrix(points)
            (_, _, _,
             t_noChange_minus_h, p_noChange_minus_h,
             t_differentiation_minus_h, p_differentiation_minus_h,
             t_integration_minus_h, p_integration_minus_h) = calculate_objective_and_plot_data(initial_dist_matrix, dist_matrix_minus_h)

            points[i, j] += h  # 恢复点的位置

            # 计算每个目标函数的梯度
            grad_t_noChange = (t_noChange_plus_h - t_noChange_minus_h) / (2 * h)
            grad_p_noChange = (p_noChange_plus_h - p_noChange_minus_h) / (2 * h)
            grad_t_differentiation = (t_differentiation_plus_h - t_differentiation_minus_h) / (2 * h)
            grad_p_differentiation = (p_differentiation_plus_h - p_differentiation_minus_h) / (2 * h)
            grad_t_integration = (t_integration_plus_h - t_integration_minus_h) / (2 * h)
            grad_p_integration = (p_integration_plus_h - p_integration_minus_h) / (2 * h)

            # 加权求和
            # 设置权重
            weights = {
                't_noChange': 10,
                'p_noChange': 1,
                't_differentiation': 1,
                'p_differentiation': 1,
                't_integration': -1,  # 注意 t_integration 的权重为负，因为我们希望它更大
                'p_integration': 1
            }
            gradient[i, j] = (weights['t_noChange'] * grad_t_noChange +
                             weights['p_noChange'] * grad_p_noChange +
                             weights['t_differentiation'] * grad_t_differentiation +
                             weights['p_differentiation'] * grad_p_differentiation +
                             weights['t_integration'] * grad_t_integration +
                             weights['p_integration'] * grad_p_integration)

    return gradient

def move_points_randomly(points, lambda_factor):
    # 随机移动点，移动后的点位置是原始位置加上一个随机值乘以lambda_factor
    new_points = points + (np.random.rand(n, 2) - 0.5) * lambda_factor
    return new_points

# 计算初始目标函数值和散点图数据
new_points = move_points_randomly(points, lambda_factor)  # 随机移动点
new_dist_matrix = calculate_distance_matrix(new_points)  # 计算新的距离矩阵
(initial_objective, initial_x, initial_y,
 initial_t_noChange, initial_p_noChange,
 initial_t_differentiation, initial_p_differentiation,
 initial_t_integration, initial_p_integration) = calculate_objective_and_plot_data(initial_dist_matrix, new_dist_matrix)

# 优化点的位置
best_points = points.copy()  # 复制初始点集作为最佳点集的初始值
best_objective = initial_objective  # 初始化最佳目标函数值为初始值
best_t_noChange = initial_t_noChange
best_p_noChange = initial_p_noChange
best_t_differentiation = initial_t_differentiation
best_p_differentiation = initial_p_differentiation
best_t_integration = initial_t_integration
best_p_integration = initial_p_integration
best_x = initial_x
best_y = initial_y

# 存储每次迭代的目标函数值、t值和p值以绘制损失曲线
objectives = [initial_objective]
t_noChange_list = [initial_t_noChange]
p_noChange_list = [initial_p_noChange]
t_differentiation_list = [initial_t_differentiation]
p_differentiation_list = [initial_p_differentiation]
t_integration_list = [initial_t_integration]
p_integration_list = [initial_p_integration]

# 进行迭代优化
for _ in tqdm(range(iterations)):
    gradient = calculate_gradient(best_points, initial_dist_matrix)  # 计算梯度
    # new_points = best_points - lambda_factor * gradient  # 沿负梯度方向更新点的位置
    new_points = best_points - gradient/np.max(np.abs(gradient))/10  # 沿负梯度方向更新点的位置
    new_dist_matrix = calculate_distance_matrix(new_points)  # 计算新的距离矩阵
    (new_objective, new_x, new_y,
     t_noChange, p_noChange,
     t_differentiation, p_differentiation,
     t_integration, p_integration) = calculate_objective_and_plot_data(initial_dist_matrix, new_dist_matrix)  # 计算新的目标函数值和散点图数据

    # 如果新的目标函数值小于最佳目标函数值，则更新最佳点集和最佳目标函数值
    criteria = new_objective < best_objective
    # criteria = (np.abs(t_noChange) <= np.abs(best_t_noChange) and
    #             p_noChange <= best_p_noChange and
    #             t_differentiation <= best_t_differentiation and
    #             p_differentiation <= best_p_differentiation and
    #             t_integration >= best_t_integration and
    #             p_integration <= best_p_integration)
    # criteria = (np.abs(t_noChange) <= np.abs(best_t_noChange) and
    #             t_differentiation <= best_t_differentiation and
    #             t_integration >= best_t_integration)
    # criteria = t_differentiation <= best_t_differentiation
    # criteria = t_integration >= best_t_integration
    # if criteria:
    best_points = new_points.copy()
    best_objective = new_objective
    best_t_noChange = t_noChange
    best_p_noChange = p_noChange
    best_t_differentiation = t_differentiation
    best_p_differentiation = p_differentiation
    best_t_integration = t_integration
    best_p_integration = p_integration
    best_x = new_x
    best_y = new_y

    objectives.append(new_objective)
    t_noChange_list.append(t_noChange)
    p_noChange_list.append(p_noChange)
    t_differentiation_list.append(t_differentiation)
    p_differentiation_list.append(p_differentiation)
    t_integration_list.append(t_integration)
    p_integration_list.append(p_integration)


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


# 绘制t值和p值曲线
plt.figure(figsize=(15, 10))

plt.subplot(3, 2, 1)
plt.plot(t_noChange_list, label='t_noChange')
plt.title("t_noChange 曲线")
plt.xlabel("迭代次数")
plt.ylabel("t_noChange")
plt.legend()

plt.subplot(3, 2, 2)
plt.plot(p_noChange_list, label='p_noChange')
plt.title("p_noChange 曲线")
plt.xlabel("迭代次数")
plt.ylabel("p_noChange")
plt.legend()

plt.subplot(3, 2, 3)
plt.plot(t_differentiation_list, label='t_differentiation')
plt.title("t_differentiation 曲线")
plt.xlabel("迭代次数")
plt.ylabel("t_differentiation")
plt.legend()

plt.subplot(3, 2, 4)
plt.plot(p_differentiation_list, label='p_differentiation')
plt.title("p_differentiation 曲线")
plt.xlabel("迭代次数")
plt.ylabel("p_differentiation")
plt.legend()

plt.subplot(3, 2, 5)
plt.plot(t_integration_list, label='t_integration')
plt.title("t_integration 曲线")
plt.xlabel("迭代次数")
plt.ylabel("t_integration")
plt.legend()

plt.subplot(3, 2, 6)
plt.plot(p_integration_list, label='p_integration')
plt.title("p_integration 曲线")
plt.xlabel("迭代次数")
plt.ylabel("p_integration")
plt.legend()

plt.tight_layout()
plt.show()



# 修改函数，使得并非对于原本的objective = obj_noChange + obj_differentiation + obj_integration进行训练，而是分别对于以下几个目标分别进行gradient的定义，最后把各个目标函数进行加权和并且更新点的移动：t_noChange更接近0，p_noChange更小，t_differentiation更小，p_differentiation更小， t_integration更大，p_integration更小