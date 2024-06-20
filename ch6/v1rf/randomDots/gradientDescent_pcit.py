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
init_learning_rate = 1e-3
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
largerIntegrationDifferentiation = 1.3
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


def pcit(x_coactivation):
    x1, y1 = 0, 0
    x2, y2 = 0.23, -0.6
    x3, y3 = 0.5, 0.1
    x4, y4 = 1, 0.4

    # Points
    x_points = np.array([x1, x2, x3, x4]) - 1.2
    y_points = np.array([y1, y2, y3, y4]) * 0.006

    # Piecewise linear function
    def piecewise_linear(x):
        if x <= x_points[0]:
            return y_points[0]
        elif x_points[0] < x <= x_points[1]:
            return y_points[0] + (y_points[1] - y_points[0]) * (x - x_points[0]) / (x_points[1] - x_points[0])
        elif x_points[1] < x <= x_points[2]:
            return y_points[1] + (y_points[2] - y_points[1]) * (x - x_points[1]) / (x_points[2] - x_points[1])
        elif x_points[2] < x <= x_points[3]:
            return y_points[2] + (y_points[3] - y_points[2]) * (x - x_points[2]) / (x_points[3] - x_points[2])
        else:
            return y_points[3]

    # Apply the piecewise function to the input array
    return np.array([piecewise_linear(x) for x in x_coactivation])



# 定义一个函数，用于计算目标函数和散点图数据
def calculate_objective_and_plot_data(initial_dist_matrix, new_dist_matrix, plotAll=False):
    # 计算移动前后的距离矩阵的上三角元素
    m1_distanceMatrix = initial_dist_matrix[np.triu_indices(n, k=1)]
    m2_distanceMatrix = new_dist_matrix[np.triu_indices(n, k=1)]

    # 将移动前的元素取相反数作为x轴，移动后的元素与移动前的元素差作为y轴
    x_coactivation = - m1_distanceMatrix  # m 越大, 越differentiation, 越少的co-activation
    y_integration = - (m2_distanceMatrix - m1_distanceMatrix)  # m 越大, 越differentiation, 因此

    xlim_target = (min(x_coactivation), max(x_coactivation))
    ylim_target = (min(y_integration), max(y_integration))
    # polynomial = scaleTargetNMPH(xlim_target, ylim_target, plotAll=plotAll, largerIntegrationDifferentiation=largerIntegrationDifferentiation)


    # 计算目标函数值
    # y_fit = polynomial(x_coactivation)
    y_fit = pcit(x_coactivation)
    objective = np.sum((y_integration - y_fit) ** 2)

    return (objective, x_coactivation, y_integration)

def calculate_gradient(points, initial_dist_matrix, curr_iter):
    gradient = np.zeros_like(points)
    h = 1e-5  # 微小的扰动，用于计算数值梯度
    for iii in range(n):
        for jjj in range(2):  # 计算每个点的x和y方向上的梯度
            if curr_iter  == iterations - 1 and iii == n - 1 and jjj == 1:
                plotAll = True
            else:
                plotAll = False
            points[iii, jjj] += h
            dist_matrix_plus_h = calculate_distance_matrix(points)
            obj_plus_h, _, _ = calculate_objective_and_plot_data(initial_dist_matrix, dist_matrix_plus_h, plotAll=plotAll)

            points[iii, jjj] -= 2 * h
            dist_matrix_minus_h = calculate_distance_matrix(points)
            obj_minus_h, _, _ = calculate_objective_and_plot_data(initial_dist_matrix, dist_matrix_minus_h, plotAll=plotAll)

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

best_x = initial_x
best_y = initial_y

# 存储每次迭代的目标函数值、t值和p值以绘制损失曲线
objectives = [initial_objective]


# 定义学习率调度器
def cosine_annealing(epoch, total_epochs, initial_lr):
    return initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))


# 进行迭代优化
for curr_iter in tqdm(range(iterations)):
    t += 1
    gradient = calculate_gradient(best_points, initial_dist_matrix, curr_iter)  # 计算梯度

    # 更新 Adam 优化器参数
    learning_rate = cosine_annealing(t, iterations, init_learning_rate)
    m = beta1 * m + (1 - beta1) * gradient
    v = beta2 * v + (1 - beta2) * (gradient ** 2)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    new_points = best_points - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)  # 使用 Adam 更新规则更新点的位置
    new_dist_matrix = calculate_distance_matrix(new_points)  # 计算新的距离矩阵
    (new_objective, new_x, new_y) = calculate_objective_and_plot_data(initial_dist_matrix, new_dist_matrix)  # 计算新的目标函数值和散点图数据

    best_points = new_points.copy()
    best_objective = new_objective
    best_x = new_x
    best_y = new_y

    objectives.append(new_objective)

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
"""


"""
对于 calculate_objective_and_plot_data 函数， 完全删除之前的额objective的定义，以及基于mean_noChange, std_noChange,mean_differentiation, std_differentiation, mean_integration, std_integration 的gradient 的定义。
取而代之的是，根据 xlim_target = (min(x_coactivation), max(x_coactivation)) 以及 ylim_target = (min(y_integration), max(y_integration))
来调用 new_poly = scaleTargetNMPH(xlim_target, ylim_target, plotAll=False)
从而可以根据当前的 x_coactivation, y_integration 与 new_poly 的差值的平方和来计算目标函数值。
"""