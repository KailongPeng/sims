# 如果新的目标函数值小于最佳目标函数值，则更新最佳点集和最佳目标函数值
# criteria = new_objective < best_objective
# criteria = (np.abs(mean_noChange) <= np.abs(best_mean_noChange) and
#             std_noChange <= best_std_noChange and
#             mean_differentiation <= best_mean_differentiation and
#             std_differentiation <= best_std_differentiation and
#             mean_integration >= best_mean_integration and
#             std_integration <= best_std_integration)
# criteria = (np.abs(mean_noChange) <= np.abs(best_mean_noChange) and
#             mean_differentiation <= best_mean_differentiation and
#             mean_integration >= best_mean_integration)
# criteria = mean_differentiation <= best_mean_differentiation
# criteria = mean_integration >= best_mean_integration
# if criteria:


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


https://blog.csdn.net/luo3300612/article/details/88397033
数值梯度：numerical gradient

"""


import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp
from tqdm import tqdm
import numpy as np

# 设置 matplotlib 的默认字体为黑体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置随机种子
np.random.seed(42)

# 定义点的数量
n = 20
# 定义控制点随机移动距离的缩放因子
lambda_factor = 0.01
# 定义优化迭代的次数
iterations = 50  # 增加迭代次数

# 在0到1的二维平面上随机均匀分布n个点
points = np.random.rand(n, 2)

def calculate_distance_matrix(points):
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i, j] = np.linalg.norm(points[i] - points[j])
    return dist_matrix

initial_dist_matrix = calculate_distance_matrix(points)

def calculate_objective_and_plot_data(initial_dist_matrix, new_dist_matrix):
    m1_distanceMatrix = initial_dist_matrix[np.triu_indices(n, k=1)]
    m2_distanceMatrix = new_dist_matrix[np.triu_indices(n, k=1)]
    x_coactivation = -m1_distanceMatrix
    y_integration = -(m2_distanceMatrix - m1_distanceMatrix)
    sorted_indices = np.argsort(x_coactivation)
    x_sorted = x_coactivation[sorted_indices]
    y_sorted = y_integration[sorted_indices]
    third = len(x_sorted) // 3
    noChange = y_sorted[:third]
    differentiation = y_sorted[third:2 * third]
    integration = y_sorted[2 * third:]
    mean_noChange, std_noChange = np.mean(noChange), np.std(noChange)
    mean_differentiation, std_differentiation = np.mean(differentiation), np.std(differentiation)
    mean_integration, std_integration = np.mean(integration), np.std(integration)
    obj_noChange = mean_noChange**2 + std_noChange
    obj_differentiation = mean_differentiation + std_differentiation
    obj_integration = -mean_integration + std_integration
    objective = obj_noChange + obj_differentiation + obj_integration
    return objective, x_coactivation, y_integration, mean_noChange, std_noChange, mean_differentiation, std_differentiation, mean_integration, std_integration

def calculate_gradient(points, initial_dist_matrix):
    gradient = np.zeros_like(points)
    h = 1e-5
    for i in range(n):
        for j in range(2):
            points[i, j] += h
            dist_matrix_plus_h = calculate_distance_matrix(points)
            (_, _, _, mean_noChange_plus_h, std_noChange_plus_h, mean_differentiation_plus_h, std_differentiation_plus_h,
             mean_integration_plus_h, std_integration_plus_h) = calculate_objective_and_plot_data(initial_dist_matrix, dist_matrix_plus_h)
            points[i, j] -= 2 * h
            dist_matrix_minus_h = calculate_distance_matrix(points)
            (_, _, _, mean_noChange_minus_h, std_noChange_minus_h, mean_differentiation_minus_h, std_differentiation_minus_h,
             mean_integration_minus_h, std_integration_minus_h) = calculate_objective_and_plot_data(initial_dist_matrix, dist_matrix_minus_h)
            points[i, j] += h
            grad_mean_noChange = (mean_noChange_plus_h**2 - mean_noChange_minus_h**2) / (2 * h)
            grad_std_noChange = (std_noChange_plus_h - std_noChange_minus_h) / (2 * h)
            grad_mean_differentiation = (mean_differentiation_plus_h - mean_differentiation_minus_h) / (2 * h)
            grad_std_differentiation = (std_differentiation_plus_h - std_differentiation_minus_h) / (2 * h)
            grad_mean_integration = (mean_integration_plus_h - mean_integration_minus_h) / (2 * h)
            grad_std_integration = (std_integration_plus_h - std_integration_minus_h) / (2 * h)
            grad_mean_noChange /= np.linalg.norm(grad_mean_noChange) + 1e-8
            grad_std_noChange /= np.linalg.norm(grad_std_noChange) + 1e-8
            grad_mean_differentiation /= np.linalg.norm(grad_mean_differentiation) + 1e-8
            grad_std_differentiation /= np.linalg.norm(grad_std_differentiation) + 1e-8
            grad_mean_integration /= np.linalg.norm(grad_mean_integration) + 1e-8
            grad_std_integration /= np.linalg.norm(grad_std_integration) + 1e-8
            weights = {
                'mean_noChange': 0,
                'std_noChange': 0,
                'mean_differentiation': 1,
                'std_differentiation': 0,
                'mean_integration': 1,
                'std_integration': 0
            }
            gradient[i, j] = (grad_mean_noChange * weights['mean_noChange']
                              + grad_std_noChange * weights['std_noChange']
                              + grad_mean_differentiation * weights['mean_differentiation']
                              + grad_std_differentiation * weights['std_differentiation']
                              - grad_mean_integration * weights['mean_integration']
                              + grad_std_integration * weights['std_integration'])
    return gradient

def move_points_randomly(points, lambda_factor):
    new_points = points + (np.random.rand(n, 2) - 0.5) * lambda_factor
    return new_points

new_points = move_points_randomly(points, lambda_factor)
new_dist_matrix = calculate_distance_matrix(new_points)
(initial_objective, initial_x, initial_y, initial_mean_noChange, initial_std_noChange, initial_mean_differentiation, initial_std_differentiation,
 initial_mean_integration, initial_std_integration) = calculate_objective_and_plot_data(initial_dist_matrix, new_dist_matrix)

best_points = points.copy()
best_objective = initial_objective
best_mean_noChange = initial_mean_noChange
best_std_noChange = initial_std_noChange
best_mean_differentiation = initial_mean_differentiation
best_std_differentiation = initial_std_differentiation
best_mean_integration = initial_mean_integration
best_std_integration = initial_std_integration
best_x = initial_x
best_y = initial_y

objectives = [initial_objective]
mean_noChange_list = [initial_mean_noChange]
std_noChange_list = [initial_std_noChange]
mean_differentiation_list = [initial_mean_differentiation]
std_differentiation_list = [initial_std_differentiation]
mean_integration_list = [initial_mean_integration]
std_integration_list = [initial_std_integration]

for _ in tqdm(range(iterations)):
    gradient = calculate_gradient(best_points, initial_dist_matrix)
    learningRate = 1e-4  # 增加学习率
    new_points = best_points - gradient * learningRate
    new_dist_matrix = calculate_distance_matrix(new_points)
    (new_objective, new_x, new_y, mean_noChange, std_noChange, mean_differentiation, std_differentiation,
     mean_integration, std_integration) = calculate_objective_and_plot_data(initial_dist_matrix, new_dist_matrix)
    if new_objective < best_objective:  # 只有在目标函数值下降时才更新最佳点集
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

xlim = (min(points[:, 0].min(), best_points[:, 0].min())-0.1, max(points[:, 0].max(), best_points[:, 0].max())+0.1)
ylim = (min(points[:, 1].min(), best_points[:, 1].min())-0.1, max(points[:, 1].max(), best_points[:, 1].max())+0.1)

plt.subplot(2, 2, 1)
plt.xlim(xlim)
plt.ylim(ylim)

plt.subplot(2, 2, 2)
plt.xlim(xlim)
plt.ylim(ylim)

plt.subplot(2, 2, 3)
plt.scatter(initial_x, initial_y, c='blue', label='初始')
plt.title("初始目标函数散点图")
plt.xlabel("-m1")
plt.ylabel("m2 - m1")
plt.legend()
plt.axhline(y=0, color='gray', linestyle='--')

plt.subplot(2, 2, 4)
plt.scatter(best_x, best_y, c='red', label='最终')
plt.title("最终目标函数散点图")
plt.xlabel("-m1")
plt.ylabel("m2 - m1")
plt.legend()
plt.axhline(y=0, color='gray', linestyle='--')

plt.tight_layout()
plt.show()

plt.figure()
plt.plot(objectives, label='目标函数值')
plt.title("损失曲线")
plt.xlabel("迭代次数")
plt.ylabel("目标函数值")
plt.legend()
plt.show()

plt.figure(figsize=(15, 10))
plt.subplot(3, 2, 1)
plt.plot(mean_noChange_list, label='mean_noChange')
plt.title("mean_noChange 曲线, 应该接近0")
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
plt.title("mean_integration 曲线， 应该更大")
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

# new_points = best_points - lambda_factor * gradient  # 沿负梯度方向更新点的位置

# new_points = best_points - gradient / np.max(np.abs(gradient)) * learningRate  # 沿负梯度方向更新点的位置

def display_cosine_annealing():
    import numpy as np
    import matplotlib.pyplot as plt

    def cosine_annealing(epoch, total_epochs, initial_lr):
        return initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))

    # 初始化参数
    init_learning_rate = 1e-4  # 初始学习率
    total_epochs = 100  # 总共的epoch数目

    # 生成epoch列表，用于计算每个epoch对应的学习率
    epochs = np.arange(1, total_epochs + 1)

    # 应用cosine_annealing函数计算每个epoch的学习率
    lrs = [cosine_annealing(epoch, total_epochs, init_learning_rate) for epoch in epochs]

    # 使用matplotlib绘制学习率随epoch变化的曲线
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, lrs)
    plt.title('Cosine Annealing Learning Rate Schedule')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    # plt.yscale('log')  # 使用对数刻度以便更好地观察小数值的变化
    plt.grid(True)
    plt.show()




# return x_vals_transformed, y_vals_transformed

# x_vals_transformed, y_vals_transformed = scaleTargetNMPH(xlim_target, ylim_target)


# Target x and y limits
# xlim_target = (0, 5)
# ylim_target = (-5, 5)
# # xlim_target = (min(x), max(x))
# # ylim_target = (min(y), max(y))
# new_poly = scaleTargetNMPH(xlim_target, ylim_target, plotAll=False)



# def calculate_gradient(points, initial_dist_matrix):
#     gradient = np.zeros_like(points)
#     h = 1e-5  # 微小的扰动，用于计算数值梯度
#     for i in range(n):
#         for j in range(2):  # 计算每个点的x和y方向上的梯度
#             points[i, j] += h
#             dist_matrix_plus_h = calculate_distance_matrix(points)
#             (_, _, _,
#              mean_noChange_plus_h, std_noChange_plus_h,
#              mean_differentiation_plus_h, std_differentiation_plus_h,
#              mean_integration_plus_h, std_integration_plus_h) = calculate_objective_and_plot_data(initial_dist_matrix, dist_matrix_plus_h)
#
#             points[i, j] -= 2 * h
#             dist_matrix_minus_h = calculate_distance_matrix(points)
#             (_, _, _,
#              mean_noChange_minus_h, std_noChange_minus_h,
#              mean_differentiation_minus_h, std_differentiation_minus_h,
#              mean_integration_minus_h, std_integration_minus_h) = calculate_objective_and_plot_data(initial_dist_matrix, dist_matrix_minus_h)
#
#             points[i, j] += h  # 恢复点的位置
#
#             # 计算每个目标函数的梯度
#             # mean_noChange更接近0，std_noChange更大，mean_differentiation更小，std_differentiation更小， mean_integration 更大，std_integration更小
#             # 总结来说： mean_noChange更接近0，mean_integration更大
#             grad_mean_noChange = (mean_noChange_plus_h**2 - mean_noChange_minus_h**2) / (2 * h)  # 注意这里是 mean_noChange 的平方， 因为我们希望它更接近0
#             grad_std_noChange = (std_noChange_plus_h - std_noChange_minus_h) / (2 * h)
#             grad_mean_differentiation = (mean_differentiation_plus_h - mean_differentiation_minus_h) / (2 * h)
#             grad_std_differentiation = (std_differentiation_plus_h - std_differentiation_minus_h) / (2 * h)
#             grad_mean_integration = (mean_integration_plus_h - mean_integration_minus_h) / (2 * h)
#             grad_std_integration = (std_integration_plus_h - std_integration_minus_h) / (2 * h)
#
#             # 梯度归一化
#             grad_mean_noChange /= np.linalg.norm(grad_mean_noChange) + 1e-8  # 防止除以零
#             grad_std_noChange /= np.linalg.norm(grad_std_noChange) + 1e-8
#             grad_mean_differentiation /= np.linalg.norm(grad_mean_differentiation) + 1e-8
#             grad_std_differentiation /= np.linalg.norm(grad_std_differentiation) + 1e-8
#             grad_mean_integration /= np.linalg.norm(grad_mean_integration) + 1e-8
#             grad_std_integration /= np.linalg.norm(grad_std_integration) + 1e-8
#
#             # 加和归一化后的梯度
#             gradient[i, j] = (
#                     grad_mean_noChange * weights['mean_noChange']
#                     + grad_std_noChange * weights['std_noChange']
#                     + grad_mean_differentiation * weights['mean_differentiation']
#                     + grad_std_differentiation * weights['std_differentiation']
#                     - grad_mean_integration * weights['mean_integration']  # 注意 mean_integration 的符号为负，因为我们希望它更大
#                     + grad_std_integration * weights['std_integration']
#             )
#
#     return gradient
# best_mean_noChange = initial_mean_noChange
# best_std_noChange = initial_std_noChange
# best_mean_differentiation = initial_mean_differentiation
# best_std_differentiation = initial_std_differentiation
# best_mean_integration = initial_mean_integration
# best_std_integration = initial_std_integration


# mean_noChange_list = [initial_mean_noChange]
# std_noChange_list = [initial_std_noChange]
# mean_differentiation_list = [initial_mean_differentiation]
# std_differentiation_list = [initial_std_differentiation]
# mean_integration_list = [initial_mean_integration]
# std_integration_list = [initial_std_integration]


# new_points = best_points - gradient * learningRate  # 沿负梯度方向更新点的位置
# new_dist_matrix = calculate_distance_matrix(new_points)  # 计算新的距离矩阵


    # best_mean_noChange = mean_noChange
    # best_std_noChange = std_noChange
    # best_mean_differentiation = mean_differentiation
    # best_std_differentiation = std_differentiation
    # best_mean_integration = mean_integration
    # best_std_integration = std_integration


    # mean_noChange_list.append(mean_noChange)
    # std_noChange_list.append(std_noChange)
    # mean_differentiation_list.append(mean_differentiation)
    # std_differentiation_list.append(std_differentiation)
    # mean_integration_list.append(mean_integration)
    # std_integration_list.append(std_integration)

# # 设置相同的xlim和ylim
# xlim = (min(points[:, 0].min(), best_points[:, 0].min())-0.1, max(points[:, 0].max(), best_points[:, 0].max())+0.1)
# ylim = (min(points[:, 1].min(), best_points[:, 1].min())-0.1, max(points[:, 1].max(), best_points[:, 1].max())+0.1)
#
# plt.subplot(2, 2, 1)
# plt.xlim(xlim)
# plt.ylim(ylim)
#
# plt.subplot(2, 2, 2)
# plt.xlim(xlim)
# plt.ylim(ylim)





# # 绘制t值和p值曲线
# plt.figure(figsize=(15, 10))
#
# plt.subplot(3, 2, 1)
# plt.plot(mean_noChange_list, label='mean_noChange')
# plt.scatter(range(len(mean_noChange_list)), mean_noChange_list, c='red')  # 这里加上这个scatter，是为了看清楚mean_noChange的变化趋势
# plt.title(f"mean_noChange 曲线, 应该接近0; final mean_noChange: {mean_noChange_list[-1]}")  # mean_noChange更接近0，mean_integration更大
# plt.xlabel("迭代次数")
# plt.ylabel("mean_noChange")
# plt.legend()
# plt.axhline(y=0, color='gray', linestyle='--')
#
# plt.subplot(3, 2, 2)
# plt.plot(std_noChange_list, label='std_noChange')
# plt.scatter(range(len(std_noChange_list)), std_noChange_list, c='red')  # 这里加上这个scatter，是为了看清楚std_noChange的变化趋势
# plt.title(f"std_noChange 曲线; final std_noChange: {std_noChange_list[-1]}")
# plt.xlabel("迭代次数")
# plt.ylabel("std_noChange")
# plt.legend()
# plt.axhline(y=0, color='gray', linestyle='--')
#
# plt.subplot(3, 2, 3)
# plt.plot(mean_differentiation_list, label='mean_differentiation')
# plt.scatter(range(len(mean_differentiation_list)), mean_differentiation_list, c='red')  # 这里加上这个scatter，是为了看清楚mean_differentiation的变化趋势
# plt.title(f"mean_differentiation 曲线; final mean_differentiation: {mean_differentiation_list[-1]}")
# plt.xlabel("迭代次数")
# plt.ylabel("mean_differentiation")
# plt.legend()
# plt.axhline(y=0, color='gray', linestyle='--')
#
# plt.subplot(3, 2, 4)
# plt.plot(std_differentiation_list, label='std_differentiation')
# plt.scatter(range(len(std_differentiation_list)), std_differentiation_list, c='red')  # 这里加上这个scatter，是为了看清楚std_differentiation的变化趋势
# plt.title(f"std_differentiation 曲线; final std_differentiation: {std_differentiation_list[-1]}")
# plt.xlabel("迭代次数")
# plt.ylabel("std_differentiation")
# plt.legend()
# plt.axhline(y=0, color='gray', linestyle='--')
#
# plt.subplot(3, 2, 5)
# plt.plot(mean_integration_list, label='mean_integration')
# plt.scatter(range(len(mean_integration_list)), mean_integration_list, c='red')  # 这里加上这个scatter，是为了看清楚mean_integration的变化趋势
# plt.title(f"mean_integration 曲线， 应该更大; final mean_integration: {mean_integration_list[-1]}")  # mean_noChange更接近0，mean_integration更大
# plt.xlabel("迭代次数")
# plt.ylabel("mean_integration")
# plt.legend()
# plt.axhline(y=0, color='gray', linestyle='--')
#
# plt.subplot(3, 2, 6)
# plt.plot(std_integration_list, label='std_integration')
# plt.scatter(range(len(std_integration_list)), std_integration_list, c='red')  # 这里加上这个scatter，是为了看清楚std_integration的变化趋势
# plt.title(f"std_integration 曲线; final std_integration: {std_integration_list[-1]}")
# plt.xlabel("迭代次数")
# plt.ylabel("std_integration")
# plt.legend()
# plt.axhline(y=0, color='gray', linestyle='--')
#
# plt.tight_layout()
# plt.show()



    # # 根据x轴的值进行排序
    # sorted_indices = np.argsort(x_coactivation)
    # x_sorted = x_coactivation[sorted_indices]
    # y_sorted = y_integration[sorted_indices]
    #
    # # 将排序后的x轴值分为三部分：noChange, differentiation, integration
    # third = len(x_sorted) // 3
    # noChange = y_sorted[:third]
    # differentiation = y_sorted[third:2 * third]
    # integration = y_sorted[2 * third:]
    #
    # # 计算三部分的目标函数值
    # mean_noChange, std_noChange = np.mean(noChange), np.std(noChange)
    # mean_differentiation, std_differentiation = np.mean(differentiation), np.std(differentiation)
    # mean_integration, std_integration = np.mean(integration), np.std(integration)
    #
    # obj_noChange = mean_noChange**2 + std_noChange  # mean_noChange越接近0，std_noChange越小，目标函数值越小
    # obj_differentiation = mean_differentiation + std_differentiation  # mean_differentiation越小于0，std_differentiation越小，目标函数值越小
    # obj_integration = - mean_integration + std_integration  # mean_integration越大于0，std_integration越小，目标函数值越小
    #
    # # 计算总体目标函数值
    # objective = obj_noChange + obj_differentiation + obj_integration
    # return (objective, x_coactivation, y_integration,
    #         mean_noChange, std_noChange, mean_differentiation, std_differentiation, mean_integration, std_integration)
