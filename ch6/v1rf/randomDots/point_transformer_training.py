import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import random

# Set random seeds for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
random.seed(seed)

testMode = False

if 'gpfs/milgram' in os.getcwd():
    os.chdir("/gpfs/milgram/project/turk-browne/projects/sandbox/simulation/sims/ch6/v1rf/randomDots")
else:
    os.chdir(r"D:\Desktop\simulation\sims\ch6\v1rf\randomDots")

print(f"Current working directory: {os.getcwd()}")
print(f"PyTorch version: {torch.__version__}")

# Define the neural network architecture
class PointTransformer(nn.Module):
    def __init__(self):
        super(PointTransformer, self).__init__()
        self.fc1 = nn.Linear(2, 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    # input shape: (40, 2)
    # fc1: (40, 2) -> (40, 50)
    # layer2 shape: (40, 50)
    # fc2: (40, 50) -> (40, 20)
    # layer3 shape: (40, 20)
    # fc3: (40, 20) -> (40, 2)
    # output shape: (40, 2)

    # 放弃记录所有的weight changes和activations，只记录
    # 每一个curr_timepoint的最后一个epoch完成后的weight of the third layer (fc3: (40, 20) -> (40, 2)) 和
    # activations of layer 3 (shape: (40, 20) and output layer (shape: (40, 2))


# Early stopping class
class EarlyStopping:
    def __init__(self, patience=None, min_delta=None):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0

    def __call__(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
            return False

        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience

# Function to train the model and record weights and activations
def train_model(best_points_history, patience=None, min_delta=None, max_epochs=None):
    initial_points = best_points_history[0]  # shape: (40, 2)
    best_points_history = torch.tensor(best_points_history, dtype=torch.float64)
    initial_points = torch.tensor(initial_points, dtype=torch.float64)

    model = PointTransformer().double()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=0)

    final_weight = []
    final_activations_layer3 = []
    final_activations_output = []

    losses = {}
    for curr_timepoint in tqdm(range(0, best_points_history.shape[0])):
        early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
        for epoch in range(max_epochs):
            optimizer.zero_grad()
            outputs = model(initial_points)
            target = best_points_history[curr_timepoint]
            loss = criterion(outputs, target)
            if epoch == 0:
                losses[curr_timepoint] = []
            losses[curr_timepoint].append(loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step()

            if early_stopping(loss.item()):
                print(f"Early stopping at epoch {epoch} for time point {curr_timepoint}")
                if testMode:
                    # Visualize training outputs and targets
                    plt.figure(figsize=(10, 5))
                    plt.scatter(outputs[:, 0].detach().numpy(), outputs[:, 1].detach().numpy(), color='blue',
                                label='Predicted Points')
                    plt.scatter(target[:, 0].detach().numpy(), target[:, 1].detach().numpy(), color='red',
                                label='Target Points')
                    plt.title(f'Time Point {curr_timepoint}, Epoch {epoch}')
                    plt.legend()
                    plt.show()

                break
            if epoch == max_epochs - 1:
                print(f"Training completed for time point {curr_timepoint}")
                if testMode:
                    # Visualize training outputs and targets
                    plt.figure(figsize=(10, 5))
                    plt.scatter(outputs[:, 0].detach().numpy(), outputs[:, 1].detach().numpy(), color='blue',
                                label='Predicted Points')
                    plt.scatter(target[:, 0].detach().numpy(), target[:, 1].detach().numpy(), color='red',
                                label='Target Points')
                    plt.title(f'Time Point {curr_timepoint}, Epoch {epoch}')
                    plt.legend()
                    plt.show()

        # Record the final weight and activations after last epoch of each curr_timepoint
        with torch.no_grad():
            final_weight.append(model.fc3.weight.clone().detach().numpy())
            activation_layer3 = torch.relu(model.fc2(torch.relu(model.fc1(initial_points))))
            final_activations_layer3.append(activation_layer3.numpy())
            final_activations_output.append(outputs.numpy())

    for curr_timepoint in range(1, best_points_history.shape[0]):
        if testMode:
            plt.figure(figsize=(10, 5))
            plt.plot(losses[curr_timepoint], label=f'Time Point {curr_timepoint}')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Loss Curve for Time Point {curr_timepoint}')
            plt.legend()
            plt.show()

    return model, final_weight, final_activations_layer3, final_activations_output

best_points_history = np.asarray(np.load('./result/best_points_history.npy')) # shape: (101, 40, 2)
if testMode:
    best_points_history = best_points_history[0:5]
    max_epochs = 100
else:
    # best_points_history = best_points_history[0:5]
    # max_epochs = 100
    best_points_history = best_points_history
    max_epochs = 100000
model, final_weight, final_activations_layer3, final_activations_output = train_model(
    best_points_history, patience=100, min_delta=1e-30, max_epochs=max_epochs)

final_weight = np.array(final_weight)
final_activations_layer3 = np.array(final_activations_layer3)
final_activations_output = np.array(final_activations_output)
print(f"final_weight.shape={final_weight.shape}")  # (101, 2, 20)
print(f"final_activations_layer3.shape={final_activations_layer3.shape}")  # (101, 40, 20)
print(f"final_activations_output.shape={final_activations_output.shape}")  # (101, 40, 2)

# tag for time
import time
time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
np.save(f'./result/final_weight_{time}.npy', final_weight)
np.save(f'./result/final_activations_layer3_{time}.npy', final_activations_layer3)
np.save(f'./result/final_activations_output_{time}.npy', final_activations_output)
print(f"Results saved with time tag: {time}")


# synaptic level 的NMPH的分析


from scipy.stats import pearsonr
from scipy.optimize import curve_fit


def cal_resample(data=None, times=5000, return_all=False):
    """
    Perform resampling on input data.

    Parameters:
    - data: Input data.
    - times: Number of resampling iterations.
    - return_all: Whether to return all resampled means.

    Returns:
    - mean: Mean of the resampled means.
    - percentile_5: 5th percentile of the resampled means.
    - percentile_95: 95th percentile of the resampled means.
    - iter_means: List of all resampled means if return_all is True.
    """
    if data is None:
        raise ValueError("Input data must be provided.")
    if isinstance(data, list):
        data = np.asarray(data)
    iter_means = [np.nanmean(data[np.random.choice(len(data), len(data), replace=True)]) for _ in range(times)]
    mean = np.mean(iter_means)
    percentile_5 = np.percentile(iter_means, 5)
    percentile_95 = np.percentile(iter_means, 95)
    if return_all:
        return mean, percentile_5, percentile_95, iter_means
    else:
        return mean, percentile_5, percentile_95


def bar(means=None, upper=None, lower=None, ROINames=None, title=None, xLabel="", yLabel="", fontsize=50,
        setBackgroundColor=False,
        savePath=None, showFigure=True):
    import matplotlib.pyplot as plt
    # plot barplot with percentage error bar
    if type(means) == list:
        means = np.asarray(means)
    if type(upper) == list:
        upper = np.asarray(upper)
    if type(means) == list:
        lower = np.asarray(lower)

    # plt.figure(figsize=(fontsize, fontsize/2), dpi=70)
    positions = list(np.arange(len(means)))

    fig, ax = plt.subplots(figsize=(fontsize/2, fontsize/2))
    ax.bar(positions, means, yerr=[means - lower, upper - means], align='center', alpha=0.5, ecolor='black',
           capsize=10)
    if setBackgroundColor:
        ax.set_facecolor((242 / 256, 242 / 256, 242 / 256))
    ax.set_ylabel(yLabel, fontsize=fontsize)
    ax.set_xlabel(xLabel, fontsize=fontsize)
    ax.set_xticks(positions)
    ax.set_facecolor((242 / 256, 242 / 256, 242 / 256))
    # Increase y-axis tick font size
    ax.tick_params(axis='y', labelsize=fontsize)

    if ROINames is not None:
        xtick = ROINames
        ax.set_xticklabels(xtick, fontsize=fontsize, rotation=45, ha='right')
    ax.set_title(title, fontsize=fontsize)
    ax.yaxis.grid(True)
    _ = plt.tight_layout()
    if savePath is not None:
        plt.savefig(savePath)
    if showFigure:
        _ = plt.show()
    else:
        _ = plt.close()


def prepare_nn_data(diff_weight, final_activations_layer3, final_activations_output):
    unit_num_layer_3 = final_activations_layer3.shape[-1]  # 20
    unit_num_output = final_activations_output.shape[-1]  # 2

    selected_channel_ids_layer_3 = list(range(unit_num_layer_3))  # [0, 1, 2, ..., 19]
    selected_channel_ids_output = list(range(unit_num_output))  # [0, 1]

    num_dots = final_activations_layer3.shape[1]  # 40
    layer_3_activations = final_activations_layer3.reshape((-1, num_dots, unit_num_layer_3))  # (101, 40, 20)
    layer_output_activations = final_activations_output.reshape((-1, num_dots, unit_num_output))  # (101, 40, 2)

    weight_changes = diff_weight  # (100, 2, 20)

    co_activations_flatten = []
    weight_changes_flatten = []
    pair_ids = []

    for curr_channel_3_feature in tqdm(range(len(selected_channel_ids_layer_3))):
        for curr_channel_output_feature in range(len(selected_channel_ids_output)):
            # 获取对于指定的synapse前后的neuron的activation
            activation_layer_3 = layer_3_activations[:, :, curr_channel_3_feature]  # (101, 40)
            activation_output = layer_output_activations[:, :, curr_channel_output_feature]  # (101, 40)
            # 获取对于指定的synapse的weight change
            weight_change = weight_changes[:, curr_channel_output_feature, curr_channel_3_feature]  # (100,)

            # 对于每一个time point, 记录指定的synapse的weight change
            weight_changes_flatten.append(weight_change)

            # 计算co-activation
            co_activation = np.multiply(activation_layer_3, activation_output)  # (101, 40)
            # 对于每一个time point, 计算40个neuron的co-activation的平均值
            co_activation = np.mean(co_activation, axis=1)  # (101,)
            # 去掉最后一个time point, 因为weight change由于做了差值, 是没有最后一个time point的
            co_activation = co_activation[0:-1] # (100,)

            co_activations_flatten.append(co_activation)
            pair_ids.append([
                selected_channel_ids_layer_3[curr_channel_3_feature],
                selected_channel_ids_output[curr_channel_output_feature]
            ])

    return co_activations_flatten, weight_changes_flatten, pair_ids


time = "2024-06-23-19-08-35"
final_weight = np.load(f'./result/final_weight_{time}.npy')
final_activations_layer3 = np.load(f'./result/final_activations_layer3_{time}.npy')
final_activations_output = np.load(f'./result/final_activations_output_{time}.npy')

# Compute the difference along the time axis (axis=0)
diff_weight = np.diff(final_weight, axis=0)
# Check the shape of the resulting matrix
print(f"diff_weight.shape={diff_weight.shape}")  # Should print (100, 2, 20)

co_activations_flatten_, weight_changes_flatten_, pair_ids_ = prepare_nn_data(
    diff_weight, final_activations_layer3, final_activations_output)


def run_NMPH(co_activations_flatten, weight_changes_flatten, pair_ids, rows=None, cols=None, plot_fig=False):

    def cubic_fit_correlation_with_params(x, y, n_splits=10, random_state=42, return_subset=True):
        def cubic_function(_x, a, b, c, d):
            return a * _x ** 3 + b * _x ** 2 + c * _x + d

        # Function to compute correlation coefficient
        def compute_correlation(observed, predicted):
            return pearsonr(observed, predicted)[0]

        # Set random seed for reproducibility
        np.random.seed(random_state)

        # Shuffle indices for k-fold cross-validation
        indices = np.arange(len(x))
        np.random.shuffle(indices)

        # Initialize arrays to store correlation coefficients and parameters
        correlation_coefficients = []
        fitted_params = []

        for curr_split in range(n_splits):
            # Split data into training and testing sets
            split_size = len(x) // n_splits
            test_indices = indices[curr_split * split_size: (curr_split + 1) * split_size]
            train_indices = np.concatenate([indices[:curr_split * split_size], indices[(curr_split + 1) * split_size:]])

            x_train, x_test = x[train_indices], x[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            # Perform constrained cubic fit on the training data
            params, _ = curve_fit(cubic_function, x_train, y_train)

            # Predict y values on the test data
            y_pred = cubic_function(x_test, *params)

            # Compute correlation coefficient and store it
            correlation_coefficient = compute_correlation(y_test, y_pred)
            correlation_coefficients.append(correlation_coefficient)

            # Store fitted parameters
            fitted_params.append(params)

        # Average correlation coefficients and parameters across folds
        mean_correlation = np.mean(correlation_coefficients)
        mean_params = np.mean(fitted_params, axis=0)

        if return_subset:
            # Randomly choose 9% of the data for future visualization
            subset_size = 10  # int(0.09 * len(x))
            subset_indices = random.sample(range(len(x)), subset_size)
            return mean_correlation, mean_params, x[subset_indices], y[subset_indices]
        else:
            return mean_correlation, mean_params

    if plot_fig:
        if rows is None:
            rows = int(np.ceil(np.sqrt(len(co_activations_flatten))))
        if cols is None:
            cols = int(np.sqrt(len(co_activations_flatten)))

        fig, axs = plt.subplots(rows, cols, figsize=(15, 15))  # Create a subplot matrix
        from matplotlib.cm import get_cmap
        cmap = get_cmap('viridis')  # Choose a colormap (you can change 'viridis' to your preferred one)
    else:
        axs = None
        cmap = None

    mean_correlation_coefficients = []
    mean_parameters = []
    x_partials = []
    y_partials = []
    for curr_pairID in tqdm(range(len(co_activations_flatten))):
        if testMode:
            test_batch_num = 5000
            x__ = co_activations_flatten[curr_pairID][:test_batch_num]
            y__ = weight_changes_flatten[curr_pairID][:test_batch_num]
            pair_id = pair_ids[curr_pairID]
        else:
            x__ = co_activations_flatten[curr_pairID]
            y__ = weight_changes_flatten[curr_pairID]
            pair_id = pair_ids[curr_pairID]
        mean_correlation_coefficient, mean_parameter, x_partial, y_partial = cubic_fit_correlation_with_params(
            x__, y__,
            n_splits=10,
            random_state=42,
            return_subset=True
        )
        mean_correlation_coefficients.append(mean_correlation_coefficient)
        mean_parameters.append(mean_parameter)
        x_partials.append(x_partial)
        y_partials.append(y_partial)

        if plot_fig:
            row = curr_pairID // cols
            col = curr_pairID % cols

            ax = axs[row, col]  # Select the appropriate subplot

            # Color the dots based on a sequence
            sequence = np.linspace(0, 1, len(x__))  # Create a sequence of values from 0 to 1
            colors = cmap(sequence)  # Map the sequence to colors using the chosen colormap

            ax.scatter(x__, y__, s=10, c=colors)  # 's' controls the size of the points, 'c' sets the colors

            # plot curve with mean_parameter
            def cubic_function(_x, a, b, c, d):
                return a * _x ** 3 + b * _x ** 2 + c * _x + d

            # Generate points for the fitted cubic curve
            x_fit = np.linspace(min(x__), max(x__), 100)
            y_fit = cubic_function(x_fit, *mean_parameter)

            # Plot the fitted cubic curve
            ax.plot(x_fit, y_fit, label='Fitted Cubic Curve', color='red')

            # Add labels and a title to each subplot
            ax.set_title(f'pairID: {pair_id}')

            # # Hide x and y-axis ticks and tick labels
            # ax.set_xticks([])
            # ax.set_yticks([])

    if plot_fig:
        plt.tight_layout()  # Adjust subplot layout for better visualization
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()

    mean_correlation_coefficients = np.array(mean_correlation_coefficients)
    # cal_resample(mean_correlation_coefficients)
    mean, percentile_5, percentile_95 = cal_resample(mean_correlation_coefficients, times=5000)
    print(f"mean={mean}, 5%={percentile_5}, 95%={percentile_95}")
    bar([mean], [percentile_5], [percentile_95], title=f"mean={mean}, 5%={percentile_5}, 95%={percentile_95}")
    p_value = np.nanmean(mean_correlation_coefficients < 0)
    print(f"p value = {p_value}")

    # Return mean_correlation_coefficients along with recorded_data
    return mean_correlation_coefficients, np.array(mean_parameters), np.array(x_partials), np.array(y_partials)


mean_correlation_coefficients_, mean_parameters_, x_partials_, y_partials_ = run_NMPH(
    co_activations_flatten_, weight_changes_flatten_, pair_ids_, plot_fig=True)

x_partials_ = x_partials_.flatten()
y_partials_ = y_partials_.flatten()
mean_parameters_avg = np.mean(mean_parameters_, axis=0)

def plot_scatter_and_cubic(x_partials, y_partials, mean_parameters):
    def cubic_function(_x, a, b, c, d):
        print(f"a={a}, b={b}, c={c}, d={d}")
        return a * _x ** 3 + b * _x ** 2 + c * _x + d

    plt.scatter(x_partials, y_partials, label='Data Points', color='green', marker='o', s=30)
    x_fit = np.linspace(min(x_partials), max(x_partials), 100)
    y_fit = cubic_function(x_fit, *mean_parameters)
    plt.plot(x_fit, y_fit, label='Fitted Cubic Curve', color='red')
    plt.xlabel('X Partials')
    plt.ylabel('Y Partials')
    plt.legend()
    plt.show()

plot_scatter_and_cubic(x_partials_, y_partials_, mean_parameters_avg)

