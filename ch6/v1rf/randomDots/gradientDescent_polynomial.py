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
iterations = 10  # 优化迭代次数
# 定义学习率
init_learning_rate = 1e-2
# 设置权重
weights = {
    'mean_noChange': 2,
    'std_noChange': 0.1,
    'mean_differentiation': 1,
    'std_differentiation': 0.1,
    'mean_integration': 1,
    'std_integration': 0.1
}
plotAll = False

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


def scaleTargetNMPH(xlim_target, ylim_target, plotAll=False):
    import numpy as np
    import matplotlib.pyplot as plt

    # Define the polynomial
    polynomial = np.poly1d([-4.35364045e-22, 1.77010151e-18, -2.93175446e-15, 2.55840429e-12,
                            -1.28789013e-09, 3.90836813e-07, -6.86498808e-05, 4.62717390e-03,
                            -1.68093593e-01, 2.24259378e+02])

    # Original x and y limits
    xlim = (7, 967)
    ylim = (5, 436)

    # # Target x and y limits
    # xlim_target = (0, 10)
    # ylim_target = (-10, 10)

    # Generate the original x values and compute the y values
    x_vals = np.linspace(xlim[0], xlim[1], 10000)
    y_vals = polynomial(x_vals)

    # Compute scaling factors
    x_scale = (xlim_target[1] - xlim_target[0]) / (xlim[1] - xlim[0])
    y_scale = (ylim_target[1] - ylim_target[0]) / (ylim[1] - ylim[0])

    # Compute translation factors
    x_translate = xlim_target[0] - xlim[0] * x_scale
    y_translate = ylim_target[0] - ylim[0] * y_scale

    # Apply scaling and translation
    x_vals_transformed = x_vals * x_scale + x_translate
    y_vals_transformed = y_vals * y_scale + y_translate

    # Find the y value at x=0 after transformation
    y_at_zero = polynomial(0 * x_scale + x_translate) * y_scale + y_translate

    # Compute the vertical shift to make the plot pass through (0, 0)
    vertical_shift = -y_at_zero

    # Apply the vertical shift
    y_vals_transformed += vertical_shift

    if plotAll:
        # Plot the transformed data
        plt.figure(figsize=(10, 6))
        plt.scatter(x_vals_transformed, y_vals_transformed, color='blue', label='Transformed Polynomial fit', s=0.0001)
        plt.xlim(xlim_target)
        plt.ylim(ylim_target)
        plt.xlabel('X axis (transformed)')
        plt.ylabel('Y axis (transformed)')
        plt.legend()
        plt.title('Transformed Polynomial Plot (passing through origin)')
        plt.grid(True)
        plt.show()
    # return x_vals_transformed, y_vals_transformed

    # x_vals_transformed, y_vals_transformed = scaleTargetNMPH(xlim_target, ylim_target)

    # Fit a new polynomial to the transformed data
    degree = 9  # Degree of the polynomial (same as the original)
    new_poly_coeffs = np.polyfit(x_vals_transformed, y_vals_transformed, degree)
    new_poly = np.poly1d(new_poly_coeffs)

    if plotAll:
        print("New Polynomial Coefficients:")
        print(new_poly_coeffs)

    if plotAll:
        # Plot the new fitted polynomial
        plt.figure(figsize=(10, 6))
        plt.scatter(x_vals_transformed, y_vals_transformed, color='blue', label='Transformed Data', s=1)
        plt.plot(x_vals_transformed, new_poly(x_vals_transformed), color='red', label='Fitted Polynomial')
        plt.xlim(xlim_target)
        plt.ylim(ylim_target)
        plt.xlabel('X axis (transformed)')
        plt.ylabel('Y axis (transformed)')
        plt.legend()
        plt.title('Fitted Polynomial Plot')
        plt.grid(True)
        plt.show()

    return new_poly

    # Target x and y limits
    # xlim_target = (0, 5)
    # ylim_target = (-5, 5)
    # # xlim_target = (min(x), max(x))
    # # ylim_target = (min(y), max(y))
    # new_poly = scaleTargetNMPH(xlim_target, ylim_target, plotAll=False)


# 定义一个函数，用于计算目标函数和散点图数据
def calculate_objective_and_plot_data(initial_dist_matrix, new_dist_matrix):
    # 计算移动前后的距离矩阵的上三角元素
    m1_distanceMatrix = initial_dist_matrix[np.triu_indices(n, k=1)]
    m2_distanceMatrix = new_dist_matrix[np.triu_indices(n, k=1)]

    # 将移动前的元素取相反数作为x轴，移动后的元素与移动前的元素差作为y轴
    x_coactivation = - m1_distanceMatrix  # m 越大, 越differentiation, 越少的co-activation
    y_integration = - (m2_distanceMatrix - m1_distanceMatrix)  # m 越大, 越differentiation, 因此

    xlim_target = (min(x_coactivation), max(x_coactivation))
    ylim_target = (min(y_integration), max(y_integration))
    polynomial = scaleTargetNMPH(xlim_target, ylim_target, plotAll=plotAll)

    # 计算目标函数值
    y_fit = polynomial(x_coactivation)
    objective = np.sum((y_integration - y_fit) ** 2)

    return (objective, x_coactivation, y_integration)
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

def calculate_gradient(points, initial_dist_matrix):
    gradient = np.zeros_like(points)
    h = 1e-5  # 微小的扰动，用于计算数值梯度
    for iii in range(n):
        for jjj in range(2):  # 计算每个点的x和y方向上的梯度
            points[iii, jjj] += h
            dist_matrix_plus_h = calculate_distance_matrix(points)
            obj_plus_h, _, _ = calculate_objective_and_plot_data(initial_dist_matrix, dist_matrix_plus_h)

            points[iii, jjj] -= 2 * h
            dist_matrix_minus_h = calculate_distance_matrix(points)
            obj_minus_h, _, _ = calculate_objective_and_plot_data(initial_dist_matrix, dist_matrix_minus_h)

            points[iii, jjj] += h  # 恢复点的位置

            # 计算目标函数的梯度
            gradient[iii, jjj] = (obj_plus_h - obj_minus_h) / (2 * h)

    return gradient



def move_points_randomly(points, lambda_factor):
    # 随机移动点，移动后的点位置是原始位置加上一个随机值乘以lambda_factor
    new_points = points + (np.random.rand(n, 2) - 0.5) * lambda_factor
    return new_points


# 初始化 Adam 优化器参数
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
m = np.zeros_like(points)  # 一阶矩估计
v = np.zeros_like(points)  # 二阶矩估计
t = 0

# 计算初始目标函数值和散点图数据
new_points = move_points_randomly(points, lambda_factor)  # 随机移动点
new_dist_matrix = calculate_distance_matrix(new_points)  # 计算新的距离矩阵
# return (objective, x_coactivation, y_integration)
(initial_objective, initial_x, initial_y) = calculate_objective_and_plot_data(initial_dist_matrix, new_dist_matrix)

# 优化点的位置
best_points = points.copy()  # 复制初始点集作为最佳点集的初始值
best_objective = initial_objective  # 初始化最佳目标函数值为初始值
# best_mean_noChange = initial_mean_noChange
# best_std_noChange = initial_std_noChange
# best_mean_differentiation = initial_mean_differentiation
# best_std_differentiation = initial_std_differentiation
# best_mean_integration = initial_mean_integration
# best_std_integration = initial_std_integration
best_x = initial_x
best_y = initial_y

# 存储每次迭代的目标函数值、t值和p值以绘制损失曲线
objectives = [initial_objective]
# mean_noChange_list = [initial_mean_noChange]
# std_noChange_list = [initial_std_noChange]
# mean_differentiation_list = [initial_mean_differentiation]
# std_differentiation_list = [initial_std_differentiation]
# mean_integration_list = [initial_mean_integration]
# std_integration_list = [initial_std_integration]


# 定义学习率调度器
def cosine_annealing(epoch, total_epochs, initial_lr):
    return initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))


# 进行迭代优化
for _ in tqdm(range(iterations)):
    t += 1
    gradient = calculate_gradient(best_points, initial_dist_matrix)  # 计算梯度

    # 更新 Adam 优化器参数
    learning_rate = cosine_annealing(t, iterations, init_learning_rate)
    m = beta1 * m + (1 - beta1) * gradient
    v = beta2 * v + (1 - beta2) * (gradient ** 2)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    new_points = best_points - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)  # 使用 Adam 更新规则更新点的位置
    new_dist_matrix = calculate_distance_matrix(new_points)  # 计算新的距离矩阵
    # new_points = best_points - gradient * learningRate  # 沿负梯度方向更新点的位置
    # new_dist_matrix = calculate_distance_matrix(new_points)  # 计算新的距离矩阵
    (new_objective, new_x, new_y) = calculate_objective_and_plot_data(initial_dist_matrix, new_dist_matrix)  # 计算新的目标函数值和散点图数据

    best_points = new_points.copy()
    best_objective = new_objective
    # best_mean_noChange = mean_noChange
    # best_std_noChange = std_noChange
    # best_mean_differentiation = mean_differentiation
    # best_std_differentiation = std_differentiation
    # best_mean_integration = mean_integration
    # best_std_integration = std_integration
    best_x = new_x
    best_y = new_y

    objectives.append(new_objective)
    # mean_noChange_list.append(mean_noChange)
    # std_noChange_list.append(std_noChange)
    # mean_differentiation_list.append(mean_differentiation)
    # std_differentiation_list.append(std_differentiation)
    # mean_integration_list.append(mean_integration)
    # std_integration_list.append(std_integration)


# 绘制初始和最终的点集位置
plt.figure(figsize=(12, 12))

plt.subplot(2, 2, 1)
plt.scatter(points[:, 0], points[:, 1], c='blue', label='初始')
# annotate each dot
for i in range(n):
    plt.annotate(f'{i}', (points[i, 0], points[i, 1]), fontsize=12, color='blue')
# plt.title("初始位置")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.legend()

# plt.subplot(2, 2, 1)
plt.scatter(best_points[:, 0], best_points[:, 1], c='red', label='最终')
# annotate each dot
for i in range(n):
    plt.annotate(f'{i}', (best_points[i, 0], best_points[i, 1]), fontsize=12, color='red')
plt.title("初始位置-最终位置")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()

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

# 绘制初始和最终的目标函数散点图
plt.subplot(2, 2, 3)
plt.scatter(initial_x, initial_y, c='blue', label='初始')
plt.title("初始目标函数散点图")
plt.xlabel("co-activation: -m1")
plt.ylabel("integration: -(m2 - m1)")
plt.legend()
# 画一条 y=0 的参考线
plt.axhline(y=0, color='gray', linestyle='--')

plt.subplot(2, 2, 4)
plt.scatter(best_x, best_y, c='red', label='最终')
plt.title("最终目标函数散点图")
plt.xlabel("co-activation: -m1")
plt.ylabel("integration: -(m2 - m1)")
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

def cubic_analysis(xAxisData, learning_effect_reshaped):
    from sklearn.model_selection import KFold

    def cal_resample(data=None, times=5000, return_all=False):
        # 这个函数的目的是为了针对输入的数据，进行有重复的抽取5000次，然后记录每一次的均值，最后输出这5000次重采样的均值分布    的   均值和5%和95%的数值。
        if data is None:
            raise Exception
        if type(data) == list:
            data = np.asarray(data)
        iter_mean = []
        for _ in range(times):
            iter_distri = data[np.random.choice(len(data), len(data), replace=True)]
            iter_mean.append(np.nanmean(iter_distri))
        _mean = np.mean(iter_mean)
        _5 = np.percentile(iter_mean, 5)
        _95 = np.percentile(iter_mean, 95)
        if return_all:
            return _mean, _5, _95, iter_mean
        else:
            return _mean, _5, _95

    # Define a cubic curve function
    def cubic_curve(x, a, b, c, d):
        return a * x ** 3 + b * x ** 2 + c * x + d

    # Fit the cubic curve to the data
    from scipy.optimize import curve_fit

    params, _ = curve_fit(cubic_curve, xAxisData, learning_effect_reshaped)

    # Generate y-values for the fitted curve over a range of x-values
    x_fit = np.linspace(xAxisData.min(), xAxisData.max(), 100)
    y_fit = cubic_curve(x_fit, *params)
    plt.plot(x_fit, y_fit, color='crimson', linewidth=3, label='Cubic Fit')  # 增加线宽，改变颜色
    plt.scatter(xAxisData, learning_effect_reshaped, color='grey', label='Data Points')  # 增加颜色

    plt.show()

    # 5折交叉验证函数
    def cross_validate_cubic_fit(x_data, y_data):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        correlation_scores = []

        for train_index, test_index in kf.split(x_data):
            x_train, x_test = x_data[train_index], x_data[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index]

            # 拟合模型
            params, _ = curve_fit(cubic_curve, x_train, y_train)

            # 在测试集上评估模型
            y_pred = cubic_curve(x_test, *params)
            corr = np.corrcoef(y_pred, y_test)[0, 1]
            correlation_scores.append(corr)
        _mean, _5, _95 = cal_resample(correlation_scores)

        return _mean, _5, _95


    # 执行5折交叉验证
    _mean, _5, _95 = cross_validate_cubic_fit(xAxisData, learning_effect_reshaped)
    yerr = np.array([[_mean - _5], [_95 - _mean]]).reshape(2, 1)

    # 画出带有误差线的条状图
    plt.figure(figsize=(10, 6))
    plt.bar(x='Mean', height=_mean, yerr=yerr, capsize=10, color='grey',
            edgecolor='black', linewidth=1.5)
    # plt.xlabel('Error Bar')
    # remove x ticks
    plt.xticks([])
    plt.ylabel('Correlation between Predicted and Observed Values')
    plt.title('5-Fold Cross-Validation of Cubic Fit')
    plt.margins(y=0.1)
    plt.show()

    # 画出binning analysis的图
    def binning_analysis_with_resample(x, y):
        x_min, x_max = np.min(x), np.max(x)
        range_x = x_max - x_min
        bin_width = range_x / 10
        total_bins = int(np.ceil(range_x / (bin_width / 9)))  # 计算总的bins数量

        # 初始化列表来存储每个bin的中值、y的均值、5%和95%置信区间
        bin_middles = []
        y_means = []
        conf_lower = []
        conf_upper = []

        # 计算bins的统计数据
        for i in range(total_bins):
            bin_start = x_min + (bin_width / 9) * i
            bin_end = bin_start + bin_width
            mask = (x >= bin_start) & (x < bin_end)
            x_current = x[mask]
            y_current = y[mask]

            if len(y_current) > 0:
                bin_middle = (bin_start + bin_end) / 2
                mean, lower, upper = cal_resample(y_current, times=500, return_all=False)  # 使用500次而不是5000次以提高性能

                bin_middles.append(bin_middle)
                y_means.append(mean)
                conf_lower.append(lower)
                conf_upper.append(upper)

        # 绘制线图
        plt.figure(figsize=(10, 6))  # 设置图表大小
        plt.plot(bin_middles, y_means, 'r', label='Mean')  # 红色线表示均值
        plt.plot(bin_middles, conf_lower, 'gray', linestyle='--', label='5% percentile')  # 灰色线表示5%置信区间
        plt.plot(bin_middles, conf_upper, 'gray', linestyle='--', label='95% percentile')  # 灰色线表示95%置信区间
        plt.fill_between(bin_middles, conf_lower, conf_upper, color='gray', alpha=0.2)  # 填充5%和95%置信区间之间的区域
        # 在y=0处添加绿色虚线
        plt.axhline(y=0, color='green', linestyle='--', alpha=0.5)  # 绿色虚线，一定的透明度

        plt.xlabel('X bin middle')
        plt.ylabel('Y resampled statistics')
        plt.title('Binning Analysis with Resampled Confidence Intervals')
        plt.legend()
        plt.show()

    binning_analysis_with_resample(xAxisData, learning_effect_reshaped)


cubic_analysis(best_x, best_y)




"""
    首先获取rep NMPH的图的x和y的lim, 然后根据这个来定义一个 正宗的NMPH curve, 然后再根据这个NMPH curve和当前的x和y的 square of difference来计算目标函数值.
    然后再根据这个目标函数值来进行优化。
    
    =
    
"""


"""
对于 calculate_objective_and_plot_data 函数， 完全删除之前的额objective的定义，以及基于mean_noChange, std_noChange,mean_differentiation, std_differentiation, mean_integration, std_integration 的gradient 的定义。
取而代之的是，根据 xlim_target = (min(x_coactivation), max(x_coactivation)) 以及 ylim_target = (min(y_integration), max(y_integration))
来调用 new_poly = scaleTargetNMPH(xlim_target, ylim_target, plotAll=False)
从而可以根据当前的 x_coactivation, y_integration 与 new_poly 的差值的平方和来计算目标函数值。
"""