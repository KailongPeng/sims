"""
Two ways to solve the current issue:
    One is to change the training and testing paradigm to be train some first test train a lot and then test
    The other way is to drastically increase the number of trials in the test probes.
        Create a python GUI so that I can draw in a 12x12 matrix whose values are 0 or 1. The save the matrix for further use.
        Create a simple GUI for drawing in a 12x12 matrix and saving the matrix values (0 or 1).
        Create a simple way for me to draw in a 12x12 matrix and saving the matrix values (0 or 1).


"""

"""
I have four matrices whose shape is (2, 196). Their names are v1actm_H, v1actm_L, v1actm_V, and v1actm_R, corresponding to the stimuli H, L, V, and R, respectively.
(2, 196) means that each matrix has 2 rows and 196 columns, corresponding to 2 trials, one before learning and one after learning.
196 is the number of neurons in the V1 region.

Now I will use correlation to compare the similarity between each stimuli H, L, V, and R.
This way for before learning and after learning, I will two 4 x 4 matrices, each of which will represent the similarity between the stimuli representations.
Now use the difference between the two matrices (after minus before) as a measure of learning or y axis of the learning curve, in other words, difference matrix is 4x4, reshape it to 1x16 and plot it as the y axis of the learning curve. 
Use the before learning matrix as the x axis of the learning curve. In other words, reshape the before learning matrix to 1x16 and plot it as the x axis of the learning curve.
Now plot the learning curve.

code the whole process in a function called `rep_NMPH` whose input is v1actm_H, v1actm_L, v1actm_V, and v1actm_R.
"""

def visualize_prob():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # Load the TSV file
    # file_path = '/gpfs/milgram/scratch60/turk-browne/kp578/chanales/v1rf/probes.tsv'
    file_path = "/gpfs/milgram/scratch60/turk-browne/kp578/chanales/v1rf/probes_new.tsv"
    probes_df = pd.read_csv(file_path, sep='\t')

    # Function to extract and reshape matrix data
    def extract_and_reshape(matrix_data, prefix):
        reshaped_matrix = np.zeros((12, 12))
        for i in range(12):
            for j in range(12):
                column_name = f'{prefix}[2:{i},{j}]'
                if column_name in matrix_data:
                    reshaped_matrix[i, j] = matrix_data[column_name]
        return reshaped_matrix

    # Prepare a dictionary to hold matrices for each category and type (%LGNon and %LGNoff)
    matrices = {}

    # Define categories and types
    categories = probes_df["$Name"].to_list()  #['H', 'L', 'V', 'R']
    types = ['%LGNon', '%LGNoff']

    # Iterate over categories and types to extract and store matrices
    for category in categories:
        matrices[category] = {}
        for type_ in types:
            category_data = probes_df[probes_df['$Name'] == category]
            if not category_data.empty:
                category_dict = category_data.iloc[0].to_dict()
                matrices[category][type_] = extract_and_reshape(category_dict, type_)

    # # Visualize each matrix pair (%LGNon and %LGNoff) side by side for each category
    # fig, axes = plt.subplots(len(categories), 2, figsize=(10, 10*len(categories)))

    # for i, category in enumerate(categories):
    #     fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    #     for j, type_ in enumerate(types):
    #         axes[j].matshow(matrices[category][type_], cmap='viridis')
    #         axes[j].set_title(f'{category} - {type_}')
    #     plt.show()
    # Visualize each matrix pair (%LGNon and %LGNoff) in one figure with a 3x8 layout
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(3, 8, figure=fig, hspace=0.3, wspace=0.3)

    for i, category in enumerate(categories):
        for j, type_ in enumerate(types):
            ax = fig.add_subplot(gs[i // 4, (i % 4) * 2 + j])
            cax = ax.matshow(matrices[category][type_], cmap='viridis')
            ax.set_title(f'{category} - {type_}')
            fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    plt.show()

#     修改这个代码使得所有的图都在一个figure里面, 这一个总体的figure的排列方式是3行4列, 因为总共有12个图. 其中每一个图分为两个部分, 分别是%LGNon和%LGNoff, 也就是一共有24个图.


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


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

# Load the dataset
path_name = r"D:\Desktop\simulation\sims\ch6\v1rf\IT_data\round2_6timepoints"
# path_name = r"D:\Desktop\simulation\sims\ch6\v1rf\IT_data\round3_4timepoints.csv"
# path_name = r"D:\Desktop\simulation\sims\ch6\v1rf\IT_data\round2_5timepoints"
# path_name = r"D:\Desktop\simulation\sims\ch6\v1rf\IT_data\round1_3timepoints"

df = pd.read_csv(path_name, sep='\t')

num_test_probes = len(df['$TrialName'].unique()) - 1
print(f"num_test_probes: {num_test_probes}")

# delete the first num_test_probes rows
df = df.iloc[num_test_probes:]
# reindex the dataframe
df.index = range(len(df))

stimuli_names = df['$TrialName'].unique()

# Find columns that begin with the specified prefixes
v1actm_cols = [col for col in df.columns if col.startswith("#V1ActM")]
lgnonact_cols = [col for col in df.columns if col.startswith("#LGNonAct")]
lgnoffact_cols = [col for col in df.columns if col.startswith("#LGNoffAct")]
ITactm_cols = [col for col in df.columns if col.startswith("#ITActM")]

# Merge the columns by summing their values
df['V1ActM'] = df[v1actm_cols].values.tolist()
df['LGNonAct'] = df[lgnonact_cols].values.tolist()
df['LGNoffAct'] = df[lgnoffact_cols].values.tolist()
df['ITActM'] = df[ITactm_cols].values.tolist()

# Drop the original columns
df.drop(v1actm_cols + lgnonact_cols + lgnoffact_cols, axis=1, inplace=True)

# extract the V1ActM, for the $TrialName L
def extract_columns(df, trial_name, column_name):
    ll = df[df['$TrialName'] == trial_name][column_name]
    # reindex the list of lists
    ll.index = range(len(ll))
    ll_array = np.zeros((len(ll), len(ll[0])))
    for ii, l in enumerate(ll):
        ll_array[ii, :] = l

    return ll_array

# Extract the V1ActM for the specified trial
layerName = "V1ActM"  # or "LGNoffAct" or "V1ActM" "ITActM"
stimuli_act = {}
for stimuli_name in stimuli_names:
    ll = extract_columns(df, stimuli_name, layerName)
    stimuli_act[stimuli_name] = ll

print(f"stimuli_act[stimuli_name].shape: {ll.shape}")

import numpy as np
import matplotlib.pyplot as plt


def compute_correlation_matrix(stimuli):
    """Compute the correlation matrix for a set of stimuli representations."""
    num_stimuli = len(stimuli)
    correlation_matrix = np.zeros((num_stimuli, num_stimuli))
    for i in range(num_stimuli):
        for j in range(i+1, num_stimuli):
            # Compute correlation for trial 0 (before learning) and trial 1 (after learning) separately
            correlation_matrix[i, j] = np.corrcoef(stimuli[i], stimuli[j])[0, 1]
    return correlation_matrix


def rep_NMPH(stimuli, before_learning_ID=0, after_learning_ID=1):
    # Compute correlation matrices for before and after learning
    before_learning = compute_correlation_matrix([s[before_learning_ID, :] for s in stimuli])  # First row for before learning
    after_learning = compute_correlation_matrix([s[after_learning_ID, :] for s in stimuli])  # Second row for after learning

    # Compute the difference matrix (learning effect)
    learning_effect = after_learning - before_learning

    # extract upper matrices for plotting
    before_learning_reshaped = before_learning[np.triu_indices(len(stimuli), k=1)]
    after_learning_reshaped = after_learning[np.triu_indices(len(stimuli), k=1)]
    learning_effect_reshaped = learning_effect[np.triu_indices(len(stimuli), k=1)]

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
    plt.plot(x_fit, y_fit, color='crimson', linewidth=3, label='Cubic Fit')  # 增加线宽，改变颜色

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
        plt.title('Binning Analysis with Resampled Confidence Intervals')
        plt.legend()
        plt.show()

    binning_analysis_with_resample(xAxisData, learning_effect_reshaped)


print(f"available time points: {stimuli_act['H'].shape[0]}")
rep_NMPH(list(stimuli_act.values()),
         before_learning_ID=0,
         after_learning_ID=-1)



