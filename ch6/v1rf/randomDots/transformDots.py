import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F

import os

os.chdir(r"D:\Desktop\simulation\sims\ch6\v1rf\randomDots")

def calculate_distances(points_):
    # Euclidean distance between two points
    distance_matrix = np.zeros((points_.shape[0], points_.shape[0]))
    for _i in range(len(points_)):
        for _j in range(len(points_)):
            distance_matrix[_i][_j] = np.linalg.norm(points_[_i] - points_[_j])
    return distance_matrix
    # return np.linalg.norm(_points[:, np.newaxis, :] - _points, axis=2)
    # calculate_distances(np.array([[1,1],
    #                              [1,1],
    #                              [8,8],
    #                              [0,0],
    #                              [3,4]]))


def calculate_distance_change(points_before, points_after):
    return np.abs(np.linalg.norm(points_after[:, np.newaxis] - points_before, axis=2))

# this random seed is preventing the model from training stably, thus this seed is removed.
# set random seed
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True

# Generate random 2D points

num_epochs = 60

hidden_dim = 20
num_layers = 10

points = np.asarray(np.load('./result/best_points_history.npy'))  # (np.array(best_points_history).shape=(101, 40, 2))。

set1 = points[0]
set2 = points[1]

def plot_points_with_colors(points, title, seed=123):
    np.random.seed(seed)  # Set random seed for reproducibility
    # colors = [plt.cm.rainbow(i / len(points)) for i in range(len(points))]
    colors = np.random.rand(len(points), 3)  # Generate random RGB colors for each point

    plt.scatter(points[:, 0], points[:, 1], c=colors)
    plt.title(title)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()

# Display set1 with random rainbow colors
plot_points_with_colors(set1, 'Set 1')

# Display set2 with random rainbow colors
plot_points_with_colors(set2, 'Set 2')


# Define a simple feedforward neural network
class SimpleTransformNet(nn.Module):
    def __init__(self):
        super(SimpleTransformNet, self).__init__()
        self.fc1 = nn.Linear(2, 5)  # Input layer: 2 input features, 5 hidden units
        self.activation_function = nn.Tanh()  # nn.ReLU() # ReLU activation function
        self.fc2 = nn.Linear(5, 5)  # Hidden layer: 5 hidden units, 1 output feature
        self.fc3 = nn.Linear(5, 2)  # Output layer: 5 hidden units, 2 output features

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation_function(x)
        x = self.fc2(x)
        x = self.activation_function(x)
        x = self.fc3(x)
        return x

# model = SimpleTransformNet()

# Function to transform points using the neural network
def transform_points(net, points):
    with torch.no_grad():
        input_tensor = torch.FloatTensor(points)
        output_tensor = net(input_tensor)
    return output_tensor.numpy()

# Function to plot the loss curve
def plot_loss_curve(loss_values, title='Training Loss Curve'):
    plt.plot(loss_values, label='Training Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Generate a neural network
# model = SimpleTransformNet()
class FlexibleTransformNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(FlexibleTransformNet, self).__init__()
        self.layers = nn.ModuleList()

        # Add input layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.ReLU())

        # Add hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())

        # Add output layer
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        residual = None
        # Apply each layer
        for currLayer, layer in enumerate(self.layers):
            if currLayer == 1:
                # Initial input for the residual connection
                residual = x

            x = layer(x)

            # Add residual connection after each ReLU
            if isinstance(layer, nn.ReLU) and currLayer < len(self.layers) - 1 and currLayer > 1:
                x = x + residual
                residual = x

        return x

input_dim = 2
output_dim = 2

model = FlexibleTransformNet(input_dim, hidden_dim, output_dim, num_layers)

class SharedWeightResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, shared_weights=None):
        super(SharedWeightResidualBlock, self).__init__()

        if shared_weights is None:
            self.shared_weights = nn.Parameter(torch.randn(input_dim, hidden_dim))
        else:
            self.shared_weights = shared_weights

        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = F.linear(x, self.shared_weights.t())  # Transpose for the first linear layer
        x = self.relu(x)
        x = F.linear(x, self.shared_weights)  # No need to transpose for the second linear layer
        return x + residual

class SharedWeightResidualTransformNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_blocks=10):
        super(SharedWeightResidualTransformNet, self).__init__()

        self.shared_weights = nn.Parameter(torch.randn(input_dim, hidden_dim))

        self.blocks = nn.ModuleList()

        # Add input block
        self.blocks.append(SharedWeightResidualBlock(input_dim, hidden_dim, self.shared_weights))

        # Add residual blocks
        for _ in range(num_blocks):
            self.blocks.append(SharedWeightResidualBlock(hidden_dim, hidden_dim, self.shared_weights))

        # Add output layer
        self.blocks.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

# input_dim = 2
# hidden_dim = 5
# output_dim = 2
# num_blocks = 10
# model = SharedWeightResidualTransformNet(input_dim, hidden_dim, output_dim, num_blocks)

criterion = nn.MSELoss()
initial_learning_rate = 0.05
optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate, weight_decay=0.001)  # Adding weight decay

# Convert sets to PyTorch tensors
set1_tensor = torch.FloatTensor(set1)
set2_tensor = torch.FloatTensor(set2)

# Display set1 to initial untrained
untrained_transformed_set1 = transform_points(model, set1)
plot_points_with_colors(untrained_transformed_set1, 'set1 to initial untrained')

# Training loop for untrained network
loss_values_untrained = []

for epoch in tqdm(range(num_epochs)):
    if epoch == int(num_epochs / 3):
        optimizer.param_groups[0]['lr'] = initial_learning_rate/2.0
    elif epoch == int(num_epochs * 2 / 3):
        optimizer.param_groups[0]['lr'] = initial_learning_rate/4.0
    outputs = model(set1_tensor)
    loss = criterion(outputs, set1_tensor)
    # loss = criterion(outputs, set2_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_values_untrained.append(loss.item())

# Display set1 to set1
untrained_transformed_set1 = transform_points(model, set1)
plot_points_with_colors(untrained_transformed_set1, 'set1 to set1')
plot_loss_curve(loss_values_untrained, title='set1 to set1: Loss Curve')


optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate, weight_decay=0.001)  # Adding weight decay
loss_values_trained = []
for epoch in tqdm(range(num_epochs)):
    if epoch == int(num_epochs / 3):
        optimizer.param_groups[0]['lr'] = initial_learning_rate/2.0
    elif epoch == int(num_epochs * 2 / 3):
        optimizer.param_groups[0]['lr'] = initial_learning_rate/4.0
    outputs = model(set1_tensor)
    loss = criterion(outputs, set2_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_values_trained.append(loss.item())

# # Display set1 to set2
trained_transformed_set1 = transform_points(model, set1)
plot_points_with_colors(trained_transformed_set1, 'set1 to set2')
plot_loss_curve(loss_values_trained, title='set1 to set2: Loss Curve')


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


# def RNN():
#     # Define a simple recurrent neural network
#     class RecurrentTransformNet(nn.Module):
#         def __init__(self):
#             super(RecurrentTransformNet, self).__init__()
#             self.rnn = nn.RNN(2, 5, batch_first=True)  # Input size: 2, Hidden size: 5
#             self.fc = nn.Linear(5, 2)  # Output layer: 5 hidden units, 2 output features
#
#         def forward(self, x):
#             out, _ = self.rnn(x.unsqueeze(0))
#             out = self.fc(out[:, -1, :])  # Take the last output from the sequence
#             return out
#
#     # Function to transform points using the recurrent neural network
#     def transform_points_rnn(net, points):
#         with torch.no_grad():
#             input_tensor = torch.FloatTensor(points).unsqueeze(0)  # Add batch dimension
#             output_tensor = net(input_tensor)
#         return output_tensor.squeeze().numpy()
#
#     # ... (previous code remains unchanged)
#
#     # Generate a recurrent neural network
#     model_rnn = RecurrentTransformNet()
#     optimizer_rnn = optim.Adam(model_rnn.parameters(), lr=0.001, weight_decay=0.01)  # Adding weight decay
#
#     # Training loop for untrained recurrent network
#     num_epochs_untrained_rnn = 1000
#     for epoch in tqdm(range(num_epochs_untrained_rnn)):
#         outputs_rnn = model_rnn(set1_tensor)
#         loss_rnn = criterion(outputs_rnn, set2_tensor)
#         optimizer_rnn.zero_grad()
#         loss_rnn.backward()
#         optimizer_rnn.step()
#
#     # Display untrained transformed set1 with random rainbow colors
#     untrained_transformed_set1_rnn = transform_points_rnn(model_rnn, set1)
#     plot_points_with_colors(untrained_transformed_set1_rnn, 'Untrained Transformed Set 1 (RNN)')
#
#     # Retrain the recurrent neural network for better results
#     model_rnn = RecurrentTransformNet()
#     optimizer_rnn = optim.Adam(model_rnn.parameters(), lr=0.001)
#     for epoch in range(num_epochs_untrained_rnn, num_epochs_untrained_rnn + 1000):
#         outputs_rnn = model_rnn(set1_tensor)
#         loss_rnn = criterion(outputs_rnn, set2_tensor)
#         optimizer_rnn.zero_grad()
#         loss_rnn.backward()
#         optimizer_rnn.step()
#
#     # Display trained transformed set1 with random rainbow colors
#     trained_transformed_set1_rnn = transform_points_rnn(model_rnn, set1)
#     plot_points_with_colors(trained_transformed_set1_rnn, 'Trained Transformed Set 1 (RNN)')
#
#
# def hopfield():
#     class HopfieldNetwork:
#         def __init__(self, size):
#             self.weights = np.zeros((size, size))
#
#         def train(self, patterns):
#             for pattern in patterns:
#                 pattern = pattern.reshape(-1, 1)
#                 self.weights += np.dot(pattern, pattern.T)
#                 np.fill_diagonal(self.weights, 0)
#
#         def recall(self, input_pattern, max_iters=100):
#             for _ in range(max_iters):
#                 activation = np.dot(self.weights, input_pattern)
#                 input_pattern = np.sign(activation)
#             return input_pattern
#
#     # Function to transform points using the Hopfield network
#     def transform_points_hopfield(hopfield_net, points):
#         transformed_points = np.zeros_like(points)
#         for i in range(len(points)):
#             transformed_points[i] = hopfield_net.recall(points[i])
#         return transformed_points
#
#     # Display points with random rainbow colors
#     def plot_points_with_colors(points, title):
#         colors = [plt.cm.rainbow(i / len(points)) for i in range(len(points))]
#
#         plt.scatter(points[:, 0], points[:, 1], c=colors)
#         plt.title(title)
#         plt.xlabel('X-axis')
#         plt.ylabel('Y-axis')
#         plt.show()
#
#     # Generate a Hopfield network
#     hopfield_net = HopfieldNetwork(size=2)
#
#     # Convert sets to NumPy arrays
#     set1_np = np.array(set1)
#     set2_np = np.array(set2)
#
#     # Training loop for Hopfield network
#     hopfield_net.train(set1_np)
#
#     # Display transformed set1 using Hopfield network with random rainbow colors
#     transformed_set1_hopfield = transform_points_hopfield(hopfield_net, set1_np)
#     plot_points_with_colors(transformed_set1_hopfield, 'Transformed Set 1 (Hopfield Network)')
#
#
# def transformer():
#     class TransformerNet(nn.Module):
#         def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
#             super(TransformerNet, self).__init__()
#             self.transformer_layer = nn.Transformer(d_model=input_dim, nhead=1, num_encoder_layers=num_layers)
#             self.fc = nn.Linear(input_dim, output_dim)
#
#         def forward(self, x):
#             x = self.transformer_layer(x)
#             x = self.fc(x)
#             return x
#
#     # Function to transform points using the neural network
#     def transform_points(net, points):
#         with torch.no_grad():
#             input_tensor = torch.FloatTensor(points).unsqueeze(0)  # Add batch dimension
#             output_tensor = net(input_tensor)
#         return output_tensor.squeeze(0).numpy()
#
#     # Generate a transformer network
#     input_dim = 2
#     hidden_dim = 5
#     output_dim = 2
#     num_transformer_layers = 1
#
#     model = TransformerNet(input_dim, hidden_dim, output_dim, num_layers=num_transformer_layers)
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)  # Adding weight decay
#
#     # Convert sets to PyTorch tensors
#     set1_tensor = torch.FloatTensor(set1)
#     set2_tensor = torch.FloatTensor(set2)
#
#     # Training loop for untrained network
#     num_epochs_untrained = 1000
#     for epoch in tqdm(range(num_epochs_untrained)):
#         outputs = model(set1_tensor)
#         loss = criterion(outputs, set2_tensor)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     # Display untrained transformed set1 with random rainbow colors
#     untrained_transformed_set1 = transform_points(model, set1)
#     plot_points_with_colors(untrained_transformed_set1, 'Untrained Transformed Set 1')
#
#     # Retrain the neural network for better results
#     model = TransformerNet(input_dim, hidden_dim, output_dim, num_layers=num_transformer_layers)
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#     for epoch in range(num_epochs_untrained, num_epochs_untrained + 1000):
#         outputs = model(set1_tensor)
#         loss = criterion(outputs, set2_tensor)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     # Display trained transformed set1 with random rainbow colors
#     trained_transformed_set1 = transform_points(model, set1)
#     plot_points_with_colors(trained_transformed_set1, 'Trained Transformed Set 1')

"""

For two sets of 100 points with 2 dimensions: set1 and set2 as given in the code, note that each dot of these two sets has a corresponding dot in the other set.
Their transformation from set1 to set2 can be achieved with a simple feedforward network. Initiate and train this neural network so that it can truthfully accomplish this transformation.

Display set2 and the transformed set1 with the un-trained and trained network with random rainbow colormap


"""