import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
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
def load_and_preprocess(file_path):
    df = pd.read_csv(file_path, sep='\t')

    # 提取有用的列并转换为列表
    df['Input'] = df[[col for col in df.columns if col.startswith("#Input")]].values.tolist()
    df['Hidden'] = df[[col for col in df.columns if col.startswith("#Hidden")]].values.tolist()
    # import pdb ; pdb.set_trace()
    # 删除原始列
    df.drop([col for col in df.columns if col.startswith("#")], axis=1, inplace=True)

    return df


def load_all_time_points(base_path, time_points):
    all_data = {}
    for time_point in time_points:
        file_path = os.path.join(base_path, f"time{time_point}.csv")
        if os.path.exists(file_path):
            all_data[time_point] = load_and_preprocess(file_path)
        else:
            print(f"File {file_path} does not exist.")
    return all_data


# 示例：加载所有时间点的数据
time_points = [0, 5, 10, 16, 20, 25, 30]  # 假设这是你所有的时间点
base_path = ("/gpfs/milgram/scratch60/turk-browne/kp578/chanales/"
             "self_org/TrainGI_0.5_TestGi_0.5_MaxEpcs_100_saveEvery5EpochFor30Epochs/")
all_data = load_all_time_points(base_path, time_points)

# print(all_data[0].head())

def compute_correlation_matrix(stimuli):
    """Compute the correlation matrix for a set of stimuli representations."""
    num_stimuli = len(stimuli)
    correlation_matrix = np.zeros((num_stimuli, num_stimuli))
    for i in range(num_stimuli):
        for j in range(i+1, num_stimuli):
            # Compute correlation for trial 0 (before learning) and trial 1 (after learning) separately
            correlation_matrix[i, j] = np.corrcoef(stimuli[i], stimuli[j])[0, 1]
    return correlation_matrix


def rep_NMPH(stimuli, before_learning_ID=0, after_learning_ID=30, layerName="Hidden"):
    # Compute correlation matrices for before and after learning
    before_learning = compute_correlation_matrix([s for s in stimuli[before_learning_ID][layerName]])
    after_learning = compute_correlation_matrix([s for s in stimuli[after_learning_ID][layerName]])

    # 计算nan的个数
    nan_count_before = np.isnan(before_learning).sum()
    nan_count_after = np.isnan(after_learning).sum()
    print(f"Before learning: {nan_count_before} NaNs")
    print(f"After learning: {nan_count_after} NaNs")
    # 计算0的个数
    zero_count_before = (before_learning == 0).sum()
    zero_count_after = (after_learning == 0).sum()
    print(f"Before learning: {zero_count_before} zeros")
    print(f"After learning: {zero_count_after} zeros")

    # Compute the difference matrix (learning effect)
    learning_effect = after_learning - before_learning
    # learning_effect after_learning  before_learning 对于以上的三个 55x55 的矩阵, 去除三个矩阵的包含nan的元素, 以及对角线以下的元素, 注意三个矩阵必须同时去除相同位置的元素
    """
    Python 中计算相关系数时可能会产生NaN（Not a Number）结果。这主要发生在以下几种情况：
    
    全零序列：如果至少有一个输入序列完全由零组成，那么其标准差为零，导致分母为零，从而产生NaN。
    
    单一值序列：如果输入序列中的所有值都相同（例如，一个序列全是1），那么该序列的方差为零，导致计算相关系数时分母为零，从而产生NaN。
    
    包含NaN值：如果输入序列中包含NaN值，根据处理方式，计算结果可能为NaN。
    """
    def remove_nan_and_lower_diagonal_elements(before_learning, after_learning, learning_effect):
        # 获取上三角矩阵的索引
        i_upper = np.triu_indices_from(before_learning, k=1)

        # 获取不含 NaN 的索引
        valid_idx = ~np.isnan(before_learning[i_upper]) & ~np.isnan(after_learning[i_upper]) & ~np.isnan(
            learning_effect[i_upper])

        # 应用索引过滤
        filtered_before_learning = before_learning[i_upper][valid_idx]
        filtered_after_learning = after_learning[i_upper][valid_idx]
        filtered_learning_effect = learning_effect[i_upper][valid_idx]

        return filtered_before_learning, filtered_after_learning, filtered_learning_effect

    before_learning_reshaped, after_learning_reshaped, learning_effect_reshaped = remove_nan_and_lower_diagonal_elements(
        before_learning, after_learning, learning_effect)

    # 计算nan的个数
    nan_count_before = np.isnan(before_learning_reshaped).sum()
    nan_count_after = np.isnan(after_learning_reshaped).sum()
    nan_count_effect = np.isnan(learning_effect_reshaped).sum()
    print(f"Before learning: {nan_count_before} NaNs")
    print(f"After learning: {nan_count_after} NaNs")
    print(f"Learning effect: {nan_count_effect} NaNs")

    # 计算0的个数
    zero_count_before = (before_learning_reshaped == 0).sum()
    zero_count_after = (after_learning_reshaped == 0).sum()
    zero_count_effect = (learning_effect_reshaped == 0).sum()
    print(f"reshaped Before learning: {zero_count_before} zeros")
    print(f"reshaped After learning: {zero_count_after} zeros")
    print(f"reshaped Learning effect: {zero_count_effect} zeros")

    import pdb; pdb.set_trace()
    # Plot learning curve
    plt.figure(figsize=(10, 6))
    xAxis = 'before learning'
    if xAxis == 'before learning':
        xAxisData = before_learning_reshaped
        plt.scatter(xAxisData, learning_effect_reshaped)
        plt.xlabel('Before Learning (Correlation)')
    else:
        xAxisData = after_learning_reshaped
        plt.scatter(xAxisData, learning_effect_reshaped)
        plt.xlabel('After Learning (Correlation)')
    plt.ylabel('Learning Effect (Difference in Correlation)')
    plt.title(f"Learning Curve (NMPH) - {before_learning_ID} vs {after_learning_ID}")
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
    plt.plot(x_fit, y_fit, color='red', label='Cubic Fit')
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
    plt.bar(x='Mean', height=_mean, yerr=yerr, capsize=10)
    plt.xlabel('Error Bar')
    plt.ylabel('Correlation between Predicted and Observed Values')
    plt.title('5-Fold Cross-Validation of Cubic Fit')
    plt.show()


print(f"available time points: {time_points}")
rep_NMPH(
    all_data,
    layerName = "Hidden",
    before_learning_ID=0,  # time_points = [0, 5, 10, 16, 20, 25, 30]  # 假设这是你所有的时间点
    after_learning_ID=30)



