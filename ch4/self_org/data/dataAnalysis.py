import pandas as pd
import numpy as np
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


# Load the dataset
# path_name = "/gpfs/milgram/scratch60/turk-browne/kp578/chanales/v1rf/record0304_1_.csv"
# path_name = "/gpfs/milgram/scratch60/turk-browne/kp578/chanales/v1rf/record0305_epc100trl100.csv"
path_name = "/gpfs/milgram/scratch60/turk-browne/kp578/chanales/v1rf/record0305_probes14_100epoch.csv"
df = pd.read_csv(path_name, sep='\t')

num_test_probes = len(df['$TrialName'].unique()) - 1

# delete the first num_test_probes rows
df = df.iloc[num_test_probes:]
# reindex the dataframe
df.index = range(len(df))

stimuli_names = df['$TrialName'].unique()

# Find columns that begin with the specified prefixes
v1actm_cols = [col for col in df.columns if col.startswith("#V1ActM")]
lgnonact_cols = [col for col in df.columns if col.startswith("#LGNonAct")]
lgnoffact_cols = [col for col in df.columns if col.startswith("#LGNoffAct")]

# Merge the columns by summing their values
df['V1ActM'] = df[v1actm_cols].values.tolist()
df['LGNonAct'] = df[lgnonact_cols].values.tolist()
df['LGNoffAct'] = df[lgnoffact_cols].values.tolist()

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
layerName = "V1ActM"  # or "LGNoffAct" or "V1ActM"
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


print(f"available time points: {stimuli_act['H'].shape[0]}")
rep_NMPH(list(stimuli_act.values()),
         before_learning_ID=0,
         after_learning_ID=8)



