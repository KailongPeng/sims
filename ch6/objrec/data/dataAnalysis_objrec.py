import os
import json
import pandas as pd
from tqdm import tqdm

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

def load_json_data_to_df(base_path, timepoints):
    # 初始化用于存储所有数据的列表
    all_data = []

    # 遍历每个时间点
    for timepoint in tqdm(timepoints, desc="Loading Data"):
        current_path = os.path.join(base_path, f"timepoint_{timepoint}")

        # 检查路径是否存在
        if not os.path.exists(current_path):
            print(f"Path does not exist: {current_path}")
            continue

        # 遍历目录中的每个文件
        for filename in os.listdir(current_path):
            if filename.endswith('.json'):
                file_path = os.path.join(current_path, filename)

                # 读取并解析 JSON 文件
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    # 为每个数据项添加time_point信息
                    data_item = {
                        'activity': data['activity'],
                        'layer_name': data['layer_name'],
                        'obj': data['obj'],
                        'trial_idx': data['trial_idx'],
                        'trial_name': data['trial_name'],
                        'time_point': timepoint  # 添加时间点信息
                    }
                    all_data.append(data_item)  # 将数据项添加到列表中

    # 将数据转换为 pandas DataFrame
    df = pd.DataFrame(all_data)

    return df

# 基路径和时间点范围
base_path = '/gpfs/milgram/scratch60/turk-browne/kp578/chanales/objrec/data'
timepoints = range(0, 6)  # 时间点从0到5

# 加载数据并转换为 DataFrame
df = load_json_data_to_df(base_path, timepoints)

# 打印 DataFrame 的信息以及前几行数据作为检查
# print(df.info())
# print(df.head())

# 现在您可以通过 df[df['layer_name'] == 'V1'] 等方式访问特定层的数据

# 筛选layer_name为'V1'且time_point为0的行，然后按trial_idx排序
t=df[df['layer_name'] == 'V1']
filtered_sorted = t[t['time_point'] == 0].sort_values(by='trial_idx')


"""
每次分析一个层，从IT开始，
首先对于df，提取出所有IT层的行，
然后对于指定的两个time_point，比如0和5，提取出两个dataframe，分别对应于IT层time_point0和 IT层time_point1.
然后这两个dataframe根据obj列进行排序.
将排好序的dataframe进行reindex.
然后获得两个dataframe的activity这一列, 转换成 list.  
现在有两个list, 分别对应于IT层time_point0和 IT层time_point1, list的长度应该是 20(测试的时候的刺激数量), 每一个list的元素是一个IT层的所有unit的激活activity.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def extract_activities_by_layer_and_timepoints(df, layer_name, timepoint1, timepoint2):
    """
    从DataFrame中提取指定层在两个时间点的活动数据。

    参数:
    df: 原始的DataFrame。
    layer_name: 字符串，指定的层名称。
    timepoint1: 整数，第一个时间点。
    timepoint2: 整数，第二个时间点。

    返回:
    activities_timepoint1: 第一个时间点的活动数据列表。
    activities_timepoint2: 第二个时间点的活动数据列表。
    """
    # 提取指定层的行
    df_layer = df[df['layer_name'] == layer_name]

    # 分别提取两个时间点的数据
    df_layer_timepoint1 = df_layer[df_layer['time_point'] == timepoint1]
    df_layer_timepoint2 = df_layer[df_layer['time_point'] == timepoint2]

    # 根据 obj 列进行排序并重新索引
    df_layer_timepoint1_sorted = df_layer_timepoint1.sort_values(by='obj').reset_index(drop=True)
    df_layer_timepoint2_sorted = df_layer_timepoint2.sort_values(by='obj').reset_index(drop=True)

    # 提取 activity 列，转换成 list
    activities_timepoint1 = df_layer_timepoint1_sorted['activity'].tolist()
    activities_timepoint2 = df_layer_timepoint2_sorted['activity'].tolist()

    return activities_timepoint1, activities_timepoint2


def compute_correlation_matrix(stimuli):
    """Compute the correlation matrix for a set of stimuli representations."""
    num_stimuli = len(stimuli)
    correlation_matrix = np.zeros((num_stimuli, num_stimuli))
    for i in range(num_stimuli):
        for j in range(i+1, num_stimuli):
            # Compute correlation for trial 0 (before learning) and trial 1 (after learning) separately
            correlation_matrix[i, j] = np.corrcoef(stimuli[i], stimuli[j])[0, 1]
    return correlation_matrix


def rep_NMPH(activities_timepoint1, activities_timepoint2, timepoint1, timepoint2, layer_name):
    # Compute correlation matrices for before and after learning
    before_learning = compute_correlation_matrix(activities_timepoint1)  # First row for before learning
    after_learning = compute_correlation_matrix(activities_timepoint2)  # Second row for after learning

    # Compute the difference matrix (learning effect)
    learning_effect = after_learning - before_learning

    # extract upper matrices for plotting
    before_learning_reshaped = before_learning[np.triu_indices(len(activities_timepoint1), k=1)]
    after_learning_reshaped = after_learning[np.triu_indices(len(activities_timepoint1), k=1)]
    learning_effect_reshaped = learning_effect[np.triu_indices(len(activities_timepoint1), k=1)]

    # import pdb;pdb.set_trace()

    # Plot learning curve
    plt.figure(figsize=(10, 6))  # 设置图表大小
    xAxis = 'before learning'
    if xAxis == 'before learning':
        xAxisData = before_learning_reshaped
        plt.scatter(xAxisData, learning_effect_reshaped,
                    color='grey', alpha=0.7, edgecolors='w', s=100,
                    label='Data Points')  # 增加标签，改变点的颜色和大小
        plt.xlabel('Before Learning (Correlation)')
    else:
        xAxisData = after_learning_reshaped
        plt.scatter(xAxisData, learning_effect_reshaped,
                    color='grey', alpha=0.7, edgecolors='w', s=100,
                    label='Data Points')  # 增加标签，改变点的颜色和大小
        plt.xlabel('After Learning (Correlation)')
    plt.ylabel('Learning Effect (Difference in Correlation)')
    plt.title(f"Learning Curve (NMPH) - {timepoint1} vs. {timepoint2} - {layer_name} Layer")
    # plt.grid(True)

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

    plt.show()

    # 5折交叉验证函数
    def cross_validate_cubic_fit(x_data, y_data):
        from sklearn.model_selection import KFold
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
    plt.title(f'5-Fold Cross-Validation of Cubic Fit - {layer_name} Layer')
    plt.margins(y=0.1)
    plt.show()

    # 画出binning analysis的图
    """
    给出一个函数， 输入是x和y， 输出是x和y的binning analysis的结果
    每一个bin的x的值是这个bin的中值，y的值是这个bin的均值和标准差
    每一个bin的宽度是所有x的range的1/10
    第一个bin的位置在所有x的最小值的位置,第二个bin的位置在第一个bin的位置加上bin的宽度的1/9，以此类推
    最后画出这个binning analysis的结果，也就是每一个bin的中值和均值和标准差，结果应该是一个bar plot。
    """

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
        plt.title(f'Binning Analysis with Resampled Confidence Intervals - {layer_name} Layer')
        plt.legend()
        plt.show()

    binning_analysis_with_resample(xAxisData, learning_effect_reshaped)


print(f"available time points: {df['time_point'].unique()}")
print(f"available layer names: {df['layer_name'].unique()}")

for layer_name in df['layer_name'].unique():
    timepoint1 = 1
    timepoint2 = 5
    activities_timepoint1, activities_timepoint2 = extract_activities_by_layer_and_timepoints(df, layer_name, timepoint1, timepoint2)
    print(f"np.asarray(activities_timepoint1).shape: {np.asarray(activities_timepoint1).shape}")
    print(f"np.asarray(activities_timepoint2).shape: {np.asarray(activities_timepoint2).shape}")

    rep_NMPH(activities_timepoint1, activities_timepoint2, timepoint1, timepoint2, layer_name)
