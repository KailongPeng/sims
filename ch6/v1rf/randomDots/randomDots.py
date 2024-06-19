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
lambda_factor = 0.1  # 示例缩放因子
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

# 定义一个函数，用于随机移动点，移动距离由lambda_factor控制
def move_points_randomly(points, lambda_factor):
    # 随机移动点，移动后的点位置是原始位置加上一个随机值乘以lambda_factor
    new_points = points + (np.random.rand(n, 2) - 0.5) * lambda_factor
    return new_points

# 定义一个函数，用于随机移动指定索引的点，移动距离由lambda_factor控制
def move_point_at_index(points, index, lambda_factor):
    # 复制点集以避免修改原始点集
    new_points = points.copy()
    # 仅随机移动指定索引的点，其他点保持不变
    new_points[index] += (np.random.rand(2) - 0.5) * lambda_factor
    return new_points

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


# 计算初始目标函数值和散点图数据
new_points = move_points_randomly(points, lambda_factor)  # 随机移动点
new_dist_matrix = calculate_distance_matrix(new_points)  # 计算新的距离矩阵
# initial_objective, initial_x, initial_y = calculate_objective_and_plot_data(initial_dist_matrix, new_dist_matrix)
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
    # new_points = move_points_randomly(points, lambda_factor)  # 随机移动点
    # index = np.random.randint(n)  # 随机选择一个点的索引
    index = _ % n  # 依次选择每个点的索引

    for trial in range(1000):
        new_points = move_point_at_index(best_points, index, lambda_factor)  # 随机移动一个点
        new_dist_matrix = calculate_distance_matrix(new_points)  # 计算新的距离矩阵
        # new_objective, new_x, new_y = calculate_objective_and_plot_data(initial_dist_matrix, new_dist_matrix)  # 计算新的目标函数值和散点图数据
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
        if criteria:
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
            break

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
