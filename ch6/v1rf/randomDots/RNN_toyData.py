import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm

testMode = False


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

# Function to generate 3D scatter plot and return data
def generate_3d_scatter_plot(separator=1 / 3, display_plot=False):
    np.random.seed(42)
    points = np.random.rand(2000, 3)  # Increase the number of points to 2000

    # Define the boundaries for region division based on the separator value
    boundaries = [separator, 2 * separator]

    # Assign labels to each point based on region
    labels_x = np.digitize(points[:, 0], boundaries)
    labels_y = np.digitize(points[:, 1], boundaries)
    labels_z = np.digitize(points[:, 2], boundaries)

    # Combine the indices of the regions in each dimension to get a unique label for each point
    if separator == 1 / 3:
        labels = labels_x + 3 * labels_y + 9 * labels_z
    elif separator == 1 / 2:
        labels = labels_x + 2 * labels_y + 4 * labels_z
    else:
        raise Exception("separator should be 1/3 or 1/2")
    if display_plot:
        # Create a randomized rainbow colormap
        rainbow_colormap = ListedColormap(np.random.rand(256, 3))

        # Create a 3D scatter plot
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot with colored points based on labels using the rainbow colormap
        # scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=labels, cmap=rainbow_colormap)
        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=labels, cmap='rainbow')

        # Add colorbar to show label-color mapping
        colorbar = plt.colorbar(scatter, ax=ax, orientation='vertical')
        colorbar.set_label('Label')

        # Set plot labels
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_title('3D Scatter Plot with Region Labels')

        # Display the plot
        plt.show()

    # Return data and figure
    return points, labels


def generate_2d_scatter_plot(display_plot=True, remove_boundary_dots=False):
    # Example usage:
    # points_2d, labels_2d = generate_2d_scatter_plot(display_plot=True)
    np.random.seed(42)
    points = np.random.rand(2650, 2)  # Generate 2D points

    # Define the boundaries for region division
    boundaries = [1 / 3, 2 / 3]

    # Set the range for points to be removed around each boundary
    boundary_range = 0.03

    # Function to create a mask for points within the specified range around a boundary
    def create_boundary_mask(coord, boundary, boundary_range):
        return np.logical_or(coord < boundary - boundary_range, coord > boundary + boundary_range)

    if remove_boundary_dots:
        # Create masks for each boundary
        mask_x_lower = ~ create_boundary_mask(points[:, 0], boundaries[0], boundary_range)
        mask_x_upper = ~ create_boundary_mask(points[:, 0], boundaries[1], boundary_range)
        mask_y_lower = ~ create_boundary_mask(points[:, 1], boundaries[0], boundary_range)
        mask_y_upper = ~ create_boundary_mask(points[:, 1], boundaries[1], boundary_range)

        # Combine masks to get a mask for points within the specified range around any boundary
        mask_within_range = np.logical_or.reduce([mask_x_lower, mask_x_upper, mask_y_lower, mask_y_upper])

        # Apply the mask to remove points within the specified range around any boundary
        points_filtered = points[~mask_within_range]
    else:
        points_filtered = points

    # Assign labels to each point based on region after filtering
    labels_x = np.digitize(points_filtered[:, 0], boundaries)
    labels_y = np.digitize(points_filtered[:, 1], boundaries)
    labels = labels_x + 3 * labels_y

    if display_plot:
        # Create a randomized rainbow colormap
        rainbow_colormap = ListedColormap(np.random.rand(256, 3))

        # Create a 2D scatter plot with filtered points
        plt.figure(figsize=(8, 8))
        plt.scatter(points_filtered[:, 0], points_filtered[:, 1], c=labels, cmap='rainbow')

        # Add colorbar to show label-color mapping
        colorbar = plt.colorbar(orientation='vertical')
        colorbar.set_label('Label')

        # Set plot labels
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('2D Scatter Plot with Region Labels (Points Removed within Range)')

        # Display the plot
        plt.show()

    # Return data
    return points_filtered, labels



def trainWith_crossEntropyLoss(threeD_input=False, remove_boundary_dots=False):

    # Total number of epochs
    total_epochs = 10000

    if threeD_input is None:
        threeD_input = True

    # this random seed is preventing the model from training stably, thus this seed is removed.
    # # set random seed
    # np.random.seed(42)
    # torch.manual_seed(42)
    # torch.cuda.manual_seed(42)
    # torch.cuda.manual_seed_all(42)
    # torch.backends.cudnn.deterministic = True

    if threeD_input:
        # Call the function to get data and figure
        points_data, labels_data = generate_3d_scatter_plot(display_plot=True)
    else:
        # Call the function to get data and figure
        points_data, labels_data = generate_2d_scatter_plot(display_plot=True, remove_boundary_dots=remove_boundary_dots)

    # Split the data into training and testing sets (1000 points each)
    train_data, test_data = points_data[:1000], points_data[1000:]
    train_labels, test_labels = labels_data[:1000], labels_data[1000:]

    # Define the neural network model
    class SimpleFeedforwardNN(nn.Module):
        def __init__(self):
            super(SimpleFeedforwardNN, self).__init__()
            if threeD_input:
                self.input_layer = nn.Linear(3, 64)  # 3D input layer
            else:
                self.input_layer = nn.Linear(2, 64)  # 2D input layer

            self.hidden_layer1 = nn.Linear(64, 32)  # First hidden layer
            self.hidden_layer2 = nn.Linear(32, 2)  # Second-to-last layer is 2D

            if threeD_input:
                self.output_layer = nn.Linear(2, 27)  # Output layer, classifying into 27 categories
            else:
                self.output_layer = nn.Linear(2, 9)  # Output layer, classifying into 9 categories

        def forward(self, x):
            x = torch.relu(self.input_layer(x)) # First layer activation
            x = torch.relu(self.hidden_layer1(x)) # Second layer activation

            self.penultimate_layer_activation = self.hidden_layer2(x)

            x = torch.relu(self.hidden_layer2(x)) # Third layer activation
            x = self.output_layer(x) # Output layer activation
            return x

    # Define weight decay loss function
    def weight_decay_loss(model, weight_decay_rate=None):
        loss = 0
        for param in model.parameters():
            loss += torch.sum(param ** 2)
        return weight_decay_rate * loss

    # Define range loss function
    def range_loss(embeddings_all, initial_x_range_0, initial_y_range_0):
        current_x_range = (torch.min(embeddings_all[:, 0]), torch.max(embeddings_all[:, 0]))
        current_y_range = (torch.min(embeddings_all[:, 1]), torch.max(embeddings_all[:, 1]))

        loss = torch.mean((current_x_range[0] - initial_x_range_0[0]) ** 2 +
                          (current_x_range[1] - initial_x_range_0[1]) ** 2 +
                          (current_y_range[0] - initial_y_range_0[0]) ** 2 +
                          (current_y_range[1] - initial_y_range_0[1]) ** 2)

        return loss

    # Instantiate the neural network model
    model = SimpleFeedforwardNN()

    # Define training data for the 1000 points
    input_train = torch.tensor(train_data, dtype=torch.float32)
    labels_train = torch.tensor(train_labels, dtype=torch.long)

    # Define testing data for the 1000 points
    input_test = torch.tensor(test_data, dtype=torch.float32)
    labels_test = torch.tensor(test_labels, dtype=torch.long)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    initial_learning_rate = 0.05
    optimizer = optim.SGD(model.parameters(), lr=initial_learning_rate)
    # Lists to store training loss values
    train_loss_history = []

    # record the initial latent space
    initial_v_points = []
    initial_v_labels = []

    # record the final latent space
    final_v_points = []
    final_v_labels = []

    # Training loop
    for epoch in tqdm(range(total_epochs)):
        # Adjust learning rate if epoch passes 1/3 of the total epochs
        if epoch == int(total_epochs / 3):
            optimizer.param_groups[0]['lr'] = initial_learning_rate / 2.0
            print(f"learning rate changed to {initial_learning_rate / 2.0}")
        if epoch == int(total_epochs * 2 / 3):
            optimizer.param_groups[0]['lr'] = initial_learning_rate / 4.0
            print(f"learning rate changed to {initial_learning_rate / 4.0}")

        # Forward pass for training data
        output_train = model(input_train)
        loss_crossEntropy = criterion(output_train, labels_train)

        embeddings_all = model.penultimate_layer_activation

        if epoch == 0:
            initial_x_range_0 = (torch.min(embeddings_all[:, 0]).item(),
                                 torch.max(embeddings_all[:, 0]).item())
            initial_y_range_0 = (torch.min(embeddings_all[:, 1]).item(),
                                 torch.max(embeddings_all[:, 1]).item())
        else:
            pass

        # Compute weight decay loss
        loss_weightDecay = weight_decay_loss(model, weight_decay_rate=0.0001)

        # Compute range loss
        loss_range = range_loss(embeddings_all, initial_x_range_0, initial_y_range_0)

        # Compute total loss
        loss_train = loss_crossEntropy #+ loss_weightDecay + loss_range

        # record initial and final latent space points
        if epoch == 0:
            # record the penultimate layer activation
            initial_v_points.append(model.penultimate_layer_activation.detach().numpy())
            initial_v_labels.append(labels_train)
        if epoch == total_epochs - 1:
            # record the penultimate layer activation
            final_v_points.append(model.penultimate_layer_activation.detach().numpy())
            final_v_labels.append(labels_train)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        # Append the training loss to the history list
        train_loss_history.append(loss_train.item())

        # # Print the loss for every 10 epochs
        # if epoch % 100 == 0:
        #     print(f'Epoch {epoch}, Training Loss: {loss_train.item()}')

    # Evaluate the model on the testing dataset and calculate accuracy
    with torch.no_grad():
        output_test = model(input_test)
        _, predicted_test = torch.max(output_test, 1)
        accuracy = (predicted_test == labels_test).float().mean()
        print(f'Testing Accuracy: {accuracy.item()}')

    # Plot the learning loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(0, len(train_loss_history)), train_loss_history, label='Training Loss')
    plt.title('Learning Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot initial and final latent space points with rainbow colormap
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Initial latent space points
    initial_v_points = np.concatenate(initial_v_points, axis=0)
    initial_v_labels = np.concatenate(initial_v_labels, axis=0)
    scatter_initial = axes[0].scatter(initial_v_points[:, 0], initial_v_points[:, 1], c=initial_v_labels, cmap='rainbow', marker='o')
    axes[0].set_title('Initial Latent Space Points')
    axes[0].set_xlabel('Dimension 1')
    axes[0].set_ylabel('Dimension 2')
    fig.colorbar(scatter_initial, ax=axes[0], label='Point Index')

    # Final latent space points
    final_v_points = np.concatenate(final_v_points, axis=0)
    final_v_labels = np.concatenate(final_v_labels, axis=0)
    scatter_final = axes[1].scatter(final_v_points[:, 0], final_v_points[:, 1], c=final_v_labels, cmap='rainbow', marker='o')
    axes[1].set_title('Final Latent Space Points')
    axes[1].set_xlabel('Dimension 1')
    axes[1].set_ylabel('Dimension 2')
    fig.colorbar(scatter_final, ax=axes[1], label='Point Index')

    plt.tight_layout()
    plt.show()


# trainWith_crossEntropyLoss(threeD_input=False, remove_boundary_dots=False)  # remove_boundary_dots=True/False: Both works, but not stably, need to run from line 1 somehow.


def train_multiple_dotsNeighbotSingleBatch(
        threeD_input=None,
        remove_boundary_dots=False,
        integrationForceScale=None, # the relative force scale between integration and differentiation.
        total_epochs=50,
        range_loss_rep_shrink=None,
        num_iterations_per_batch=None,
        plot_neighborhood=False,  # whether to plot the neighborhood of each point in latent space or not):
        num_close=None,
        num_background=None,
        hidden_dim=None,
        num_layers=None,
        loss_type=None,
    ):
    """
    design the output of test_single_dotsNeighbotSingleBatch should be
        for each batch:
            layer activation (A layer = penultimate layer, B layer = last layer)
                A layer ; before training ; center points
                A layer ; before training ; close neighbors
                A layer ; before training ; background neighbors
                B layer ; before training ; center points
                B layer ; before training ; close neighbors
                B layer ; before training ; background neighbors

                A layer ; after training ; center points
                A layer ; after training ; close neighbors
                A layer ; after training ; background neighbors
                B layer ; after training ; center points
                B layer ; after training ; close neighbors
                B layer ; after training ; background neighbors
            weight change  
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    import matplotlib.pyplot as plt
    import numpy as np
    from tqdm import tqdm
    import torch.nn.init as init

    # set random seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True

    if threeD_input is None:
        threeD_input = True

    # Define batch size, number of close neighbors, and number of background neighbors
    batch_size = 1

    if integrationForceScale is None:
        integrationForceScale = 1
    # (num_close, num_background) = (5, 5)  # num_close: number of close neighbors, num_background: number of background neighbors
    """ (num_close, num_background) = (?, ?)  # num_close: number of close neighbors, num_background: number of background neighbors
        result: 
        (0, 1) gets normal result,
        (1, 1) gets normal result,
        (2, 2) gets normal result,
        (5, 5) gets normal result,
        (10, 10) gets normal result,
        (15, 15) gets collapsed result, 
        (20, 20) gets collapsed result
        
        (1, 20) gets normal result,        
        (20, 1) gets collapsed result,
        
        conclusion:
        1. the number of close neighbors and background neighbors should be small, otherwise the result will be collapsed.
        2. whether the result collapses or not most likely depends on the number of close neighbors, not the number of background neighbors.
        3. intuitively, this is determined by whether probabilistically speaking, the close neighbors of two neighboring center points overlap or not. If overlap, then the result will be collapsed.
        4. it turns out that the close neighbors are not important, the background neighbors are important. This means that integration is not important while differentiation is important for representation learning.
        5. one way to boost integration might be to increase the weight of close neighbor pulling force to enforce the formation of clusters.
    """

    print(f"integrationForceScale={integrationForceScale}")
    print(f"number of close neighbors={num_close}, number of background neighbors={num_background}")
    # assert c + b + 1 <= batch_size, f"c + b + 1 should be less than or equal to {batch_size}"

    # Define toy dataset
    class ToyDataset(Dataset):
        def __init__(self, data, labels):
            self.data = data
            self.labels = labels

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]

    def prepare_data_rnn(dataset, sequence_length=None):
        """
        Prepares the data for RNN input.

        Parameters:
        - dataset: numpy array with shape (N, 2) where N is the number of points and 2 is the number of coordinates.
        - sequence_length: Desired sequence length for the RNN input.

        Returns:
        - rnn_input: numpy array with shape (N, sequence_length, 2).
        """
        num_points, num_coordinates = dataset.shape

        # Repeat each point sequence_length times
        rnn_input = np.tile(dataset[:, np.newaxis, :], (1, sequence_length, 1))
        # axis 0: number of points, axis 1: sequence length, axis 2: number of coordinates

        return torch.tensor(rnn_input, dtype=torch.float32)

    # Define neural network architecture
    # Define the Vanilla RNN model
    class VanillaRNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(VanillaRNN, self).__init__()

            self.hidden_size = hidden_size

            # RNN layer
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)

            # Fully connected layer
            self.fc = nn.Linear(hidden_size, output_size)

            self.input_layer = self.rnn
            self.hidden_layer1 = self.rnn
            self.hidden_layer2 = self.fc  # self.hidden_layer2 is the thing that is used in synaptic NMPH analysis.

        def forward(self, x):
            # preparing data for RNN input
            num_timepoints = num_layers
            x = prepare_data_rnn(
                x,
                sequence_length=num_timepoints)

            # RNN forward pass
            out, _ = self.rnn(x)

            # Take the output from the last time step
            out = out[:, -1, :]
            self.penultimate_layer_activation = out.detach().numpy()

            # Fully connected layer
            out = self.fc(out)
            self.final_layer_activation = out.detach().numpy()

            return out

    # Define local aggregation loss function
    def local_aggregation_loss(
            embeddings_center, close_neighbors, background_neighbors, integrationForceScale=None,
    ):
        # Compute pairwise distances between embeddings and close neighbors  # embeddings(50,2) close_neighbors(50,10,2) -> (50,10)
        if close_neighbors.shape[1] == 0:
            close_distances = torch.tensor(1e-6)
        else:
            expanded_embeddings = embeddings_center.unsqueeze(1).expand_as(
                close_neighbors)  # Expand the embeddings to have the same dimensions as close_neighbors
            close_distances = torch.mean(torch.norm(expanded_embeddings - close_neighbors,
                                         dim=2))  # Calculate the Euclidean distance
        if background_neighbors.shape[1] == 0:
            background_distances = torch.tensor(1e-6)
        else:
            # Compute pairwise distances between embeddings and background neighbors
            expanded_embeddings = embeddings_center.unsqueeze(1).expand_as(
                background_neighbors)  # Expand the embeddings to have the same dimensions as background_neighbors
            background_distances = torch.mean(torch.norm(expanded_embeddings - background_neighbors,
                                              dim=2))  # Calculate the Euclidean distance

        # Compute loss based on distances
        # loss = torch.mean(torch.log(1 + torch.exp(integrationForceScale * close_distances - background_distances)))
        loss = integrationForceScale * close_distances - background_distances

        return loss
    def local_aggregation_vecterScale_loss(
            embeddings_center, close_neighbors, background_neighbors,
            close_neighbors_moved, background_neighbors_moved,
            integrationForceScale=None,
    ):
        # Reshape embeddings_center to match the shape of close_neighbors and background_neighbors
        embeddings_center = embeddings_center.unsqueeze(1)
        unmoved_dots = torch.cat((embeddings_center, close_neighbors, background_neighbors), dim=1)
        moved_dots = torch.cat((embeddings_center, close_neighbors_moved, background_neighbors_moved), dim=1)
        # Compute pairwise distances between unmoved and moved dots
        criterion = nn.MSELoss()
        loss = criterion(unmoved_dots, moved_dots)

        return loss

    # Define range loss function
    def range_loss(embeddings_all, initial_x_range_0, initial_y_range_0):
        # current_x_range = torch.max(embeddings_all[:, 0]) - torch.min(embeddings_all[:, 0])
        # current_y_range = torch.max(embeddings_all[:, 1]) - torch.min(embeddings_all[:, 1])
        #
        # loss = ((current_x_range - initial_x_range_0) ** 2 +
        #         (current_y_range - initial_y_range_0) ** 2)

        current_x_range = (torch.min(embeddings_all[:, 0]), torch.max(embeddings_all[:, 0]))
        current_y_range = (torch.min(embeddings_all[:, 1]), torch.max(embeddings_all[:, 1]))

        loss = torch.mean((current_x_range[0] - initial_x_range_0[0]) ** 2 +
                          (current_x_range[1] - initial_x_range_0[1]) ** 2 +
                          (current_y_range[0] - initial_y_range_0[0]) ** 2 +
                          (current_y_range[1] - initial_y_range_0[1]) ** 2)

        return loss

    def get_close_and_background_neighbors(
            center_embedding, all_embeddings, num_close=None, num_background=None,
            move_factor=None
    ):
        """
        Get close and background neighbors for each point in center_embeddings based on distances to all_embeddings.

        Args:
            center_embedding (torch.Tensor): Embeddings of center points.
            all_embeddings (torch.Tensor): Embeddings of all points.
            num_close (int): Number of close neighbors to retrieve for each center point.
            num_background (int): Number of background neighbors to retrieve for each center point.

        Returns:
            torch.Tensor: Embeddings of close neighbors.
            torch.Tensor: Embeddings of background neighbors.
            torch.Tensor: Indices of center points.
            torch.Tensor: Indices of close neighbors.
            torch.Tensor: Indices of background neighbors.
        """
        # Compute pairwise distances between center_embeddings and all_embeddings
        distances = torch.cdist(center_embedding, all_embeddings)  # (1, 2) (1000, 2) -> (1, 1000)  # This step is taking longer time.

        # Get indices of k closest neighbors for each example
        _, indices = torch.topk(distances, num_close + num_background + 1, largest=False)

        center_indices = indices[:, 0]

        # Remove self from list of neighbors
        indices = indices[:, 1:]

        # # Get indices of center points
        # center_indices = torch.arange(center_embeddings.size(0)).view(-1, 1).expand(-1, num_close + num_background)

        # Get embeddings of close neighbors
        close_neighbors = all_embeddings[indices[:, :num_close]]
        close_indices = indices[:, :num_close]
        close_neighbors_moved = close_neighbors - move_factor * (close_neighbors - center_embedding)

        # Get embeddings of background neighbors
        background_neighbors = all_embeddings[indices[:, num_close:]]
        background_indices = indices[:, num_close:]
        background_neighbors_moved = background_neighbors + move_factor * (background_neighbors - center_embedding)

        return (close_neighbors, background_neighbors, close_indices, background_indices, center_indices,
                close_neighbors_moved, background_neighbors_moved)

    # Define toy dataset shaped [1000, 2]
    if threeD_input:
        points_data, labels_data = generate_3d_scatter_plot(separator=1 / 2,
                                                            display_plot=True)  # separator should be 1/3 or 1/2
    else:
        points_data, labels_data = generate_2d_scatter_plot(display_plot=True, remove_boundary_dots=remove_boundary_dots)


    # Split the data into training and testing sets (1000 points each)
    train_data, test_data = points_data[:1000], points_data[1000:]

    train_labels, test_labels = labels_data[:1000], labels_data[1000:]

    dataset = ToyDataset(train_data, train_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define neural network and optimizer
    # net = SimpleFeedforwardNN(threeD_input=threeD_input,
    #                           init_zero_weights=False,
    #                           use_batch_norm=False, use_layer_norm=False)  # It was found that layer norm is much better than batch norm.
    # net = SimpleRNNWithActivations(input_size=2, hidden_size=5, output_size=2)

    # input_dim = 2
    # hidden_dim = hidden_dim  # 20
    # output_dim = 2
    # num_layers = num_layers  # 5
    # net = FlexibleTransformNet(input_dim, hidden_dim, output_dim, num_layers)

    # Instantiate the model
    input_size = 2  # Size of input features
    hidden_size = hidden_dim  # Size of hidden state
    output_size = 2  # Size of output
    net = VanillaRNN(input_size, hidden_size, output_size)


    learning_rate = 0.05 * 4
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.001)

    # Train network using local aggregation loss
    loss_values = []  # List to store loss values for each epoch

    # record the initial latent space
    initial_v_points = []
    initial_v_labels = []

    # record the final latent space
    final_v_points = []
    final_v_labels = []

    # Record the weights, penultimate layer activations, and final layer activations
    weight_difference_history = {'input_layer': [], 'hidden_layer1': [], 'hidden_layer2': []}

    activation_history = {
        'A layer ; before training ; center points': [],
        'A layer ; before training ; close neighbors': [],
        'A layer ; before training ; background neighbors': [],

        'A layer ; after training ; center points': [],
        'A layer ; after training ; close neighbors': [],
        'A layer ; after training ; background neighbors': [],

        'B layer ; before training ; center points': [],
        'B layer ; before training ; close neighbors': [],
        'B layer ; before training ; background neighbors': [],

        'B layer ; after training ; center points': [],
        'B layer ; after training ; close neighbors': [],
        'B layer ; after training ; background neighbors': []}
    initial_x_range_0 = None
    initial_y_range_0 = None
    ax = None
    print(f"num_iterations_per_batch={num_iterations_per_batch}")
    for epoch in tqdm(range(total_epochs)):
        if epoch == int(total_epochs / 3):
            optimizer.param_groups[0]['lr'] = learning_rate / 2.0
            print(f"learning rate changed to {learning_rate / 2.0}")
        if epoch == int(total_epochs * 2 / 3):
            optimizer.param_groups[0]['lr'] = learning_rate / 4.0
            print(f"learning rate changed to {learning_rate / 4.0}")
        epoch_loss = 0.0  # Variable to accumulate loss within each epoch

        # for curr_batch, (batch, batch_labels) in enumerate(dataloader):
        for curr_batch, (batch, batch_labels) in tqdm(enumerate(dataloader)):
            # Record weights
            input_layer_before = net.input_layer.weight_ih_l0.data.clone().detach().numpy() # (20,2)  # rnn input to hidden
            # net.input_layer.bias_ih_l0.data.clone().detach().numpy()  # (20,)
            hidden_layer1_before = net.hidden_layer1.weight_hh_l0.data.clone().detach().numpy() # (20,20)  # rnn hidden to hidden
            # net.hidden_layer1.bias_hh_l0.data.clone().detach().numpy()  # (20,)
            hidden_layer2_before = net.hidden_layer2.weight.data.clone().detach().numpy()  # (2,20)  # rnn hidden to output

            close_neighbors_moved, background_neighbors_moved = None, None
            for iteration in range(num_iterations_per_batch):  # Introduce a loop for multiple weight update iterations
                # record initial and final latent space points
                if epoch == 0 and curr_batch == 0 and iteration == 0:
                    # record the penultimate layer activation
                    _ = net(torch.tensor(train_data, dtype=torch.float32))
                    initial_v_points = net.final_layer_activation
                    initial_v_labels = train_labels
                    # initial_v_points.append(embeddings_ceterPoint)
                    # initial_v_labels.append(batch_labels)

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                # embedding_centerPoint = net(batch.float())  # embeddings of the current batch, note that batch_size=1, so this is a single center point.
                # embeddings_all = net(torch.tensor(train_data, dtype=torch.float32))  # embeddings of all points

                embedding_centerPoint = net(batch.float())  # embeddings of the current batch, note that batch_size=1, so this is a single center point.
                embeddings_all = net(train_data)  # embeddings of all points



                # Get close and background neighbors
                if iteration == 0:
                    (close_neighbors, background_neighbors, close_indices, background_indices, center_indices,
                     close_neighbors_moved, background_neighbors_moved) = get_close_and_background_neighbors(
                        embedding_centerPoint,  # single center point when batch_size=1
                        embeddings_all,  # embeddings of all points
                        num_close=num_close,
                        num_background=num_background,
                        move_factor=0.5  # move_factor = [0-1]
                    )
                    close_neighbors_moved = close_neighbors_moved.detach().numpy()
                    background_neighbors_moved = background_neighbors_moved.detach().numpy()
                else:
                    (close_neighbors, background_neighbors, close_indices, background_indices, center_indices,
                     _, _) = get_close_and_background_neighbors(
                        embedding_centerPoint,  # single center point when batch_size=1
                        embeddings_all,  # embeddings of all points
                        num_close=num_close,
                        num_background=num_background,
                        move_factor=0.3
                    )

                # record activations
                activation_history['A layer ; before training ; center points'].append(net.penultimate_layer_activation[center_indices])
                activation_history['A layer ; before training ; close neighbors'].append(net.penultimate_layer_activation[close_indices])
                activation_history['A layer ; before training ; background neighbors'].append(net.penultimate_layer_activation[background_indices])
                activation_history['B layer ; before training ; center points'].append(net.final_layer_activation[center_indices])
                activation_history['B layer ; before training ; close neighbors'].append(net.final_layer_activation[close_indices])
                activation_history['B layer ; before training ; background neighbors'].append(net.final_layer_activation[background_indices])

                if plot_neighborhood and iteration == 0:
                    assert batch_size == 1  # assert that there is only one center point, otherwise the plot will be messy.
                    # plot the current center point(red cross) and its close (blue dots) and background (black dots) neighbors in latent space
                    latent_points = embeddings_all.detach().numpy()
                    fig = plt.figure(figsize=(20, 20))
                    ax = fig.add_subplot(111)

                    # other points
                    ax.scatter(latent_points[:, 0], latent_points[:, 1], c='gray', marker='o',
                               alpha=0.5)
                    # close neighbors
                    ax.scatter(latent_points[close_indices, 0], latent_points[close_indices, 1],
                                c='blue', marker='o', label='Close before', alpha=0.5)
                    # background neighbors
                    ax.scatter(latent_points[background_indices, 0], latent_points[background_indices, 1],
                                c='black', marker='o', label='Background before', alpha=0.5)
                    # current center point
                    ax.scatter(latent_points[center_indices, 0], latent_points[center_indices, 1], c='red', marker='*',
                                s=200, label='Center before', alpha=0.5)

                    # plot close_neighbors_moved, background_neighbors_moved
                    close_neighbors_moved_points = close_neighbors_moved  #.detach().numpy()
                    background_neighbors_moved_points = background_neighbors_moved  #.detach().numpy()
                    ax.scatter(close_neighbors_moved_points[0, :, 0], close_neighbors_moved_points[0, :, 1],
                                c='cyan', marker='o', label='Close target', alpha=0.5)
                    ax.scatter(background_neighbors_moved_points[0, :, 0], background_neighbors_moved_points[0, :, 1],
                                c='red', marker='o', label='Background target', alpha=0.5)

                # update weight begin
                if epoch == 0 and curr_batch == 0 and iteration == 0:
                    if range_loss_rep_shrink is not None:
                        print(f"rep_shrink={range_loss_rep_shrink}")
                        # initial_x_range_0 = rep_shrink * (torch.max(embeddings_all[:, 0]).item() -
                        #                                   torch.min(embeddings_all[:, 0]).item())
                        # initial_y_range_0 = rep_shrink * (torch.min(embeddings_all[:, 1]).item() -
                        #                                   torch.max(embeddings_all[:, 1]).item())

                        initial_x_range_0 = (
                            range_loss_rep_shrink * torch.min(embeddings_all[:, 0]).item(),
                            range_loss_rep_shrink * torch.max(embeddings_all[:, 0]).item()
                        )
                        initial_y_range_0 = (
                            range_loss_rep_shrink * torch.min(embeddings_all[:, 1]).item(),
                            range_loss_rep_shrink * torch.max(embeddings_all[:, 1]).item()
                        )
                else:
                    pass

                if loss_type == 'push_pull_loss':
                    # Call local_aggregation_loss, weight_decay_loss, and range_loss functions
                    loss_local_aggregation  = local_aggregation_loss(
                        embedding_centerPoint,
                        close_neighbors, background_neighbors,
                        integrationForceScale=integrationForceScale
                    )
                elif loss_type == 'vector_scale_loss':
                    loss_local_aggregation = local_aggregation_vecterScale_loss(
                        embedding_centerPoint,
                        close_neighbors, background_neighbors,
                        # close_neighbors_moved, background_neighbors_moved
                        torch.tensor(close_neighbors_moved), torch.tensor(background_neighbors_moved),
                        # integrationForceScale=integrationForceScale
                    )
                else:
                    raise ValueError(f"loss_type={loss_type} is not supported.")

                if range_loss_rep_shrink is None:
                    loss_range = 0
                else:
                    loss_range = range_loss(
                        embeddings_all=embeddings_all,  # Provide actual embeddings_all
                        initial_x_range_0=initial_x_range_0,  # Provide actual initial_x_range_0
                        initial_y_range_0=initial_y_range_0  # Provide actual initial_y_range_0
                    )

                # Combine losses
                total_loss = loss_local_aggregation + loss_range #+ loss_weight_decay
                if plot_neighborhood:
                    print(f"loss_local_aggregation={loss_local_aggregation}")
                    print(f"loss_range={loss_range}")

                # Backward pass
                total_loss.backward()
                # Update parameters
                optimizer.step()

                epoch_loss += loss_local_aggregation.item()

                # Record weights
                latent_points = net(torch.tensor(train_data, dtype=torch.float32)).detach().numpy()
                input_layer_after = net.input_layer.weight_ih_l0.data.clone().detach().numpy()  # (20,2)  # rnn input to hidden
                # net.input_layer.bias_ih_l0.data.clone().detach().numpy()  # (20,)
                hidden_layer1_after = net.hidden_layer1.weight_hh_l0.data.clone().detach().numpy()  # (20,20)  # rnn hidden to hidden
                # net.hidden_layer1.bias_hh_l0.data.clone().detach().numpy()  # (20,)
                hidden_layer2_after = net.hidden_layer2.weight.data.clone().detach().numpy()  # (2,20)  # rnn hidden to output

                weight_difference_history['input_layer'].append(input_layer_after - input_layer_before)
                weight_difference_history['hidden_layer1'].append(hidden_layer1_after - hidden_layer1_before)
                weight_difference_history['hidden_layer2'].append(hidden_layer2_after - hidden_layer2_before)

                # record activations
                activation_history['A layer ; after training ; center points'].append(net.penultimate_layer_activation[center_indices])
                activation_history['A layer ; after training ; close neighbors'].append(net.penultimate_layer_activation[close_indices])
                activation_history['A layer ; after training ; background neighbors'].append(net.penultimate_layer_activation[background_indices])
                activation_history['B layer ; after training ; center points'].append(net.final_layer_activation[center_indices])
                activation_history['B layer ; after training ; close neighbors'].append(net.final_layer_activation[close_indices])
                activation_history['B layer ; after training ; background neighbors'].append(net.final_layer_activation[background_indices])

                # record initial and final latent space points
                if epoch == total_epochs - 1 and curr_batch == len(dataloader) - 1 and iteration == num_iterations_per_batch - 1:
                    # record the penultimate layer activation
                    _ = net(torch.tensor(train_data, dtype=torch.float32))
                    final_v_points = net.final_layer_activation
                    final_v_labels = train_labels
                    # final_v_points.append(embeddings_ceterPoint)
                    # final_v_labels.append(batch_labels)

                if plot_neighborhood and iteration == num_iterations_per_batch - 1:
                    # other points
                    ax.scatter(latent_points[:, 0], latent_points[:, 1], c='gray', marker='o',
                               alpha=0.1)
                    # close neighbors
                    ax.scatter(latent_points[close_indices, 0], latent_points[close_indices, 1],
                                # c='green',
                               c=(0.5, 1.0, 0.5),  # light green
                               marker='o', label='Close after', alpha=1)
                    # background neighbors
                    ax.scatter(latent_points[background_indices, 0], latent_points[background_indices, 1],
                                # c='purple',
                               c=(1.0, 1.0, 0.5),  # light yellow
                               marker='o', label='Background after', alpha=1)
                    # current center point
                    ax.scatter(latent_points[center_indices, 0], latent_points[center_indices, 1], c='red', marker='*',
                                s=200, label='Center after', alpha=0.5)

                    ax.set_xlabel('X Label')
                    ax.set_ylabel('Y Label')
                    ax.set_title(f'Epoch-{epoch} iteration-{iteration} after training')  # loss={epoch_loss:.2f}
                    ax.legend()
                    plt.show()

        # Calculate average loss for the epoch
        average_epoch_loss = epoch_loss / len(dataloader)
        loss_values.append(average_epoch_loss)

        # if epoch % np.ceil(total_epochs / 3) == 0:
        #     # Print and record the average loss for the epoch
        #     print(f'Epoch [{epoch + 1}/{total_epochs}], Loss: {average_epoch_loss}')

    # Plot the loss curve
    plt.plot(range(1, len(loss_values) + 1), loss_values, marker='o')
    plt.title('Local Aggregation Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Average Loss')
    plt.grid(True)
    plt.show()

    # Plot initial and final latent space points with rainbow colormap
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Set consistent xlim and ylim for both subplots
    set_sameXY_plotting_range = True
    if set_sameXY_plotting_range:
        _min_val = (
            min(initial_v_points[:, 0].min(), final_v_points[:, 0].min()),
            min(initial_v_points[:, 1].min(), final_v_points[:, 1].min())
        )
        _max_val = (
            max(initial_v_points[:, 0].max(), final_v_points[:, 0].max()),
            max(initial_v_points[:, 1].max(), final_v_points[:, 1].max())
        )
        min_val = (_min_val[0] - (_max_val[0]-_min_val[0])/20,
                   _min_val[1] - (_max_val[1]-_min_val[1])/20)
        max_val = (_max_val[0] + (_max_val[0]-_min_val[0])/20,
                   _max_val[1] + (_max_val[1]-_min_val[1])/20)
    else:
        min_val = None
        max_val = None

    # Initial latent space points
    scatter_initial = axes[0].scatter(
        initial_v_points[:, 0], initial_v_points[:, 1],
        c=initial_v_labels,
        cmap='rainbow', marker='o')
    axes[0].set_title('Initial Latent Space Points')
    axes[0].set_xlabel('Dimension 1')
    axes[0].set_ylabel('Dimension 2')
    if set_sameXY_plotting_range:
        axes[0].set_xlim(min_val[0], max_val[0])
        axes[0].set_ylim(min_val[1], max_val[1])
    # fig.colorbar(scatter_initial, ax=axes[0], label='Point Index')

    # Final latent space points
    scatter_final = axes[1].scatter(
        final_v_points[:, 0], final_v_points[:, 1],
        c=final_v_labels,
        cmap='rainbow', marker='o')
    axes[1].set_title('Final Latent Space Points')
    axes[1].set_xlabel('Dimension 1')
    axes[1].set_ylabel('Dimension 2')
    if set_sameXY_plotting_range:
        axes[1].set_xlim(min_val[0], max_val[0])
        axes[1].set_ylim(min_val[1], max_val[1])
    # fig.colorbar(scatter_final, ax=axes[1], label='Point Index')

    plt.tight_layout()
    plt.show()

    # save the weight difference history
    weight_difference_folder = f"/gpfs/milgram/scratch60/turk-browne/kp578/LocalAgg/toyData/weight_difference_folder/"
    os.makedirs(weight_difference_folder, exist_ok=True)

    # save the weight difference history
    np.save(f'{weight_difference_folder}/weight_difference_history_input_layer.npy', np.asarray(
        weight_difference_history[
            'input_layer']))  # (20000, 64, 2)  1000points/50imagesPerBatch=20batchPerEpoch, in total there are 20*1000epoch=20000 batches
    np.save(f'{weight_difference_folder}/weight_difference_history_hidden_layer1.npy',
            np.asarray(weight_difference_history['hidden_layer1']))  # (20000, 32, 64)
    np.save(f'{weight_difference_folder}/weight_difference_history_hidden_layer2.npy',
            np.asarray(weight_difference_history['hidden_layer2']))  # (20000, 2, 32)

    # save the activation history
    np.save(f'{weight_difference_folder}/A_layer_before_training_center_points.npy',
            np.asarray(activation_history['A layer ; before training ; center points']))  # (# batch, 1, 32)
    np.save(f'{weight_difference_folder}/A_layer_before_training_close_neighbors.npy',
            np.asarray(activation_history['A layer ; before training ; close neighbors']))  # (# batch, num_close, 32)
    np.save(f'{weight_difference_folder}/A_layer_before_training_background_neighbors.npy',
            np.asarray(activation_history['A layer ; before training ; background neighbors']))  # (# batch, num_background, 32)

    np.save(f'{weight_difference_folder}/B_layer_before_training_center_points.npy',
            np.asarray(activation_history['B layer ; before training ; center points']))  # (# batch, 1, 2)
    np.save(f'{weight_difference_folder}/B_layer_before_training_close_neighbors.npy',
            np.asarray(activation_history['B layer ; before training ; close neighbors']))  # (# batch, num_close, 2)
    np.save(f'{weight_difference_folder}/B_layer_before_training_background_neighbors.npy',
            np.asarray(activation_history['B layer ; before training ; background neighbors']))  # (# batch, num_close, 2)

    np.save(f'{weight_difference_folder}/A_layer_after_training_center_points.npy',
            np.asarray(activation_history['A layer ; after training ; center points']))  # (# batch, 1, 32)
    np.save(f'{weight_difference_folder}/A_layer_after_training_close_neighbors.npy',
            np.asarray(activation_history['A layer ; after training ; close neighbors']))  # (# batch, num_close, 32)
    np.save(f'{weight_difference_folder}/A_layer_after_training_background_neighbors.npy',
            np.asarray(activation_history['A layer ; after training ; background neighbors']))  # (# batch, num_background, 32)

    np.save(f'{weight_difference_folder}/B_layer_after_training_center_points.npy',
            np.asarray(activation_history['B layer ; after training ; center points']))  # (# batch, 1, 2)
    np.save(f'{weight_difference_folder}/B_layer_after_training_close_neighbors.npy',
            np.asarray(activation_history['B layer ; after training ; close neighbors']))  # (# batch, num_close, 2)
    np.save(f'{weight_difference_folder}/B_layer_after_training_background_neighbors.npy',
            np.asarray(activation_history['B layer ; after training ; background neighbors']))  # (# batch, num_background, 2)

total_epochs = 1
num_iterations_per_batch = 5  # increase this from 1 to 10, the ratio of mean_background/mean_close increases from 1.64 to 3.09
(num_close, num_background)=(10, 10)  # after changing the local_aggregation_loss function, the ratio of mean_background/mean_close becomes extremely stably and not easy to collapse.
hidden_dim = 40
num_layers = 10
loss_type = 'push_pull_loss'  # 'push_pull_loss' 'vector_scale_loss'
train_multiple_dotsNeighbotSingleBatch(
    threeD_input=False,
    remove_boundary_dots=False,
    integrationForceScale=1,  # the relative force scale between integration and differentiation. Should be 1, 2 collapses the result
    total_epochs=total_epochs,
    range_loss_rep_shrink=None,  # rep_shrink can be None (not using range loss), 1, 0.9, 0.85, 0.8, whether range loss is implemented
    num_iterations_per_batch=num_iterations_per_batch,
    plot_neighborhood=False,
    num_close=num_close,
    num_background=num_background,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    loss_type=loss_type,
)  # this works as long as the number of close neighbors is small  # both remove_boundary_dots True and False works.  # integrationForceScale=1.5 does not work.

"""
    representational level
        for every pair of points, calculate the distance between them before and after training for each batch, 
        calculate the difference between the distances and called it the learning 
        The distance between the points in the latent space before training is recorded. The opposite of this distance is called co-activation.
        
        Then the co-activation and learning are plotted against each other as X and Y axes respectively. This should be the representational level NMPH curve.
        
    synaptic level
        for the connection between the penultimate layer and the final layer, record the weight before and after training and calculate the difference between them.
        record the activation of the penultimate layer and the final layer before and after training.
        
        plot the weight difference against the co-activation as X and Y axes respectively. This should be the synaptic level NMPH curve.  
"""

def representational_level(total_epochs=None, batch_num_per_epoch=None, num_closePoints=None, num_backgroundPoints=None):
    """

    design the input of NMPH_representational should be
        A layer ; before training ; center points
        A layer ; before training ; close neighbors
        A layer ; before training ; background neighbors
        B layer ; before training ; center points
        B layer ; before training ; close neighbors
        B layer ; before training ; background neighbors

        A layer ; after training ; center points
        A layer ; after training ; close neighbors
        A layer ; after training ; background neighbors
        B layer ; after training ; center points
        B layer ; after training ; close neighbors
        B layer ; after training ; background neighbors


    """
    import random
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr
    from scipy.optimize import curve_fit
    from tqdm import tqdm
    from matplotlib.cm import get_cmap

    testMode = True
    if testMode:
        testBatchNum = 500
    else:
        testBatchNum = None
    repChange_distanceType = 'L2'  # 'cosine', 'L1', 'L2', 'dot' 'correlation' 'jacard'(slow)
    coactivation_distanceType = 'L2'  # 'cosine', 'L1', 'L2', 'dot' 'correlation' 'jacard'(slow)
    co_activationType = 'before'  # 'before', 'after'
    num_centerPoints = 1
    # (num_closePoints, num_backgroundPoints) = (5, 5)

    # Define paths for data folders
    weight_difference_folder = "/gpfs/milgram/scratch60/turk-browne/kp578/LocalAgg/toyData/weight_difference_folder/"

    directory_path = "/gpfs/milgram/scratch60/turk-browne/kp578/LocalAgg/toyData/representational_level/"
    os.makedirs(directory_path, exist_ok=True)

    assert num_centerPoints == 1

    def binarize_representations(representations, threshold=0.1):
        percentile = np.percentile(representations, threshold * 100)
        return (representations > percentile).astype(int)

    def calculate_jaccard_similarity(representation1, representation2):
        from sklearn.metrics import jaccard_score
        bin_rep1 = binarize_representations(representation1)
        bin_rep2 = binarize_representations(representation2)
        return jaccard_score(bin_rep1, bin_rep2)

    def prepare_data():
        # Set seed
        random.seed(131)

        # Load data
        weight_difference_history_input_layer = np.load(
            f'{weight_difference_folder}/weight_difference_history_input_layer.npy')
        total_batch_num = weight_difference_history_input_layer.shape[0]
        print(f"Total Batch Num: {total_batch_num}")  # 10000

        """
            load         
                A layer ; before training ; center points
                A layer ; before training ; close neighbors
                A layer ; before training ; background neighbors
                B layer ; before training ; center points
                B layer ; before training ; close neighbors
                B layer ; before training ; background neighbors

                A layer ; after training ; center points
                A layer ; after training ; close neighbors
                A layer ; after training ; background neighbors
                B layer ; after training ; center points
                B layer ; after training ; close neighbors
                B layer ; after training ; background neighbors
        """
        # A_layer_before_training_center_points = np.load(
        #     f'{weight_difference_folder}/A_layer_before_training_center_points.npy')
        # A_layer_before_training_center_points = A_layer_before_training_center_points.reshape((50000, 1, 1, 32))
        # A_layer_before_training_close_neighbors = np.load(
        #     f'{weight_difference_folder}/A_layer_before_training_close_neighbors.npy')
        # A_layer_before_training_background_neighbors = np.load(
        #     f'{weight_difference_folder}/A_layer_before_training_background_neighbors.npy')
        # layer_a_activations_before = np.concatenate([
        #     A_layer_before_training_close_neighbors,
        #     A_layer_before_training_background_neighbors,
        #     A_layer_before_training_center_points
        # ], axis=2)
        # layer_a_activations_before = layer_a_activations_before.reshape((50000, 11, 32))

        B_layer_before_training_center_points = np.load(
            f'{weight_difference_folder}/B_layer_before_training_center_points.npy')
        B_layer_before_training_center_points = B_layer_before_training_center_points.reshape((total_epochs*batch_num_per_epoch, 1, 1, 2))

        if num_closePoints != 0:
            B_layer_before_training_close_neighbors = np.load(
                f'{weight_difference_folder}/B_layer_before_training_close_neighbors.npy')
        else:
            B_layer_before_training_close_neighbors = np.zeros((total_epochs*batch_num_per_epoch, 1, 0, 2))
        if num_backgroundPoints != 0:
            B_layer_before_training_background_neighbors = np.load(
                f'{weight_difference_folder}/B_layer_before_training_background_neighbors.npy')
            B_layer_before_training_background_neighbors = B_layer_before_training_background_neighbors.reshape((
                total_epochs*batch_num_per_epoch, 1, num_backgroundPoints, 2))
        else:
            B_layer_before_training_background_neighbors = np.zeros((total_epochs*batch_num_per_epoch, 1, 0, 2))
        # B_layer_before_training_background_neighbors = np.load(
        #     f'{weight_difference_folder}/B_layer_before_training_background_neighbors.npy')

        layer_b_activations_before = np.concatenate([
            B_layer_before_training_center_points,
            B_layer_before_training_close_neighbors,
            B_layer_before_training_background_neighbors,
        ], axis=2)

        # Remove the axis with size 1
        layer_b_activations_before = np.squeeze(layer_b_activations_before, axis=1)

        # A_layer_after_training_center_points = np.load(
        #     f'{weight_difference_folder}/A_layer_after_training_center_points.npy')
        # A_layer_after_training_center_points = A_layer_after_training_center_points.reshape((50000, 1, 1, 32))
        # A_layer_after_training_close_neighbors = np.load(
        #     f'{weight_difference_folder}/A_layer_after_training_close_neighbors.npy')
        # A_layer_after_training_background_neighbors = np.load(
        #     f'{weight_difference_folder}/A_layer_after_training_background_neighbors.npy')
        # layer_a_activations_after = np.concatenate([
        #     A_layer_after_training_close_neighbors,
        #     A_layer_after_training_background_neighbors,
        #     A_layer_after_training_center_points
        # ], axis=2)

        B_layer_after_training_center_points = np.load(
            f'{weight_difference_folder}/B_layer_after_training_center_points.npy')
        B_layer_after_training_center_points = B_layer_after_training_center_points.reshape((total_epochs*batch_num_per_epoch, 1, 1, 2))
        if num_closePoints != 0:
            B_layer_after_training_close_neighbors = np.load(
                f'{weight_difference_folder}/B_layer_after_training_close_neighbors.npy')
        else:
            B_layer_after_training_close_neighbors = np.zeros((total_epochs*batch_num_per_epoch, 1, 0, 2))
        if num_backgroundPoints != 0:
            B_layer_after_training_background_neighbors = np.load(
                f'{weight_difference_folder}/B_layer_after_training_background_neighbors.npy')
            B_layer_after_training_background_neighbors = B_layer_after_training_background_neighbors.reshape((
                total_epochs*batch_num_per_epoch, 1, num_backgroundPoints, 2))
        else:
            B_layer_after_training_background_neighbors = np.zeros((total_epochs*batch_num_per_epoch, 1, 0, 2))
        layer_b_activations_after = np.concatenate([
            B_layer_after_training_center_points,
            B_layer_after_training_close_neighbors,
            B_layer_after_training_background_neighbors,
        ], axis=2)

        # Remove the axis with size 1
        layer_b_activations_after = np.squeeze(layer_b_activations_after, axis=1)

        return layer_b_activations_before, layer_b_activations_after

    layer_b_activations_before, layer_b_activations_after = prepare_data()  # layer_b_activations_before (10000, 50, 2) layer_b_activations_after (10000, 50, 2)

    # if testMode:
    #     layer_b_activations_before = layer_b_activations_before[:testBatchNum]  # (5, 11, 2) (batch#, 1+5+5, 2 dimensions)
    #     layer_b_activations_after = layer_b_activations_after[:testBatchNum]  # (5, 11, 2) (batch#, 1+5+5, 2 dimensions)

    if not os.path.exists(f'{directory_path}/temp'):
        os.mkdir(f'{directory_path}/temp')

    # if not testMode:
    #     np.save(f'{directory_path}/temp/co_activations_flatten_.npy',
    #             co_activations_flatten_)  # shape = [pair#, batch#]
    #     np.save(f'{directory_path}/temp/weight_changes_flatten_.npy',
    #             weight_changes_flatten_)  # shape = [pair#, batch#]
    #     np.save(f'{directory_path}/temp/pairIDs_.npy',
    #             pairIDs_)  # shape = [pair#, [ID1, ID2]]

    # co_activations_flatten_ = np.load(f'{directory_path}/temp/co_activations_flatten_.npy',
    #                                   allow_pickle=True)  # shape = [pair#, batch#]
    # weight_changes_flatten_ = np.load(f'{directory_path}/temp/weight_changes_flatten_.npy',
    #                                   allow_pickle=True)  # shape = [pair#, batch#]
    # pairIDs_ = np.load(f'{directory_path}/temp/pairIDs_.npy',
    #                    allow_pickle=True)  # shape = [pair#, [ID1, ID2]]

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

    def prepare_data_for_NMPH_only_between_centerPoint(
            curr_batch_=None,
            layer_activations_before=None,
            layer_activations_after=None,
            repChange_distanceType_=None,
            coactivation_distanceType_=None):
        def isolate_axis(matrix):
            """
            Isolate the first axis of a matrix into three separate matrices.

            Parameters:
            - matrix: Input matrix with shape (11, 2) or (num_centerPoints+num_closePoints+num_backgroundPoints, 2)

            Returns:
            - center: Matrix with shape (1, 2) or (num_centerPoints, 2)
            - close: Matrix with shape (5, 2) or (num_closePoints, 2)
            - background: Matrix with shape (5, 2) or (num_backgroundPoints, 2)
            """
            center = matrix[
                     0:
                     num_centerPoints, :]
            close = matrix[
                    num_centerPoints:
                    num_centerPoints+num_closePoints, :]
            background = matrix[
                         num_centerPoints+num_closePoints:
                         num_centerPoints+num_closePoints+num_backgroundPoints, :]
            neighbor = np.concatenate([close, background], axis=0)

            return center, close, background, neighbor

        layer_activations_before = layer_activations_before[curr_batch_, :, :]  # (11, 2)
        layer_activations_after = layer_activations_after[curr_batch_, :, :]  # (11, 2)

        center_before, close_before, background_before, neighbor_before = isolate_axis(layer_activations_before)
        center_after, close_after, background_after, neighbor_after = isolate_axis(layer_activations_after)

        pairImg_similarity_before_repChange = np.zeros((
            num_centerPoints,
            num_closePoints + num_backgroundPoints))
        pairImg_similarity_before_coactivation = np.zeros((
            num_centerPoints,
            num_closePoints + num_backgroundPoints))

        def similarity_between_center_and_neighbors(center, neighbor, distanceType):
            center = center.reshape((1, -1))  # (1, 2)
            neighbor = neighbor.reshape((1, -1))  # (1, 2)
            if distanceType == 'cosine':
                return (np.dot(center, neighbor.T) / (np.linalg.norm(center) * np.linalg.norm(neighbor, axis=1))).reshape((1, -1))
            elif distanceType == 'dot':
                return (np.dot(center, neighbor.T)).reshape((1, -1))
            elif distanceType == 'correlation':
                return (pearsonr(center.reshape(-1), neighbor.reshape(-1))[0]).reshape((1, -1))
            elif distanceType == 'L1':
                return (- np.linalg.norm(center - neighbor, ord=1)).reshape((1, -1))
            elif distanceType == 'L2':  # euclidean
                return (- np.linalg.norm(center - neighbor, ord=2)).reshape((1, -1))
            elif distanceType == 'jacard':
                return calculate_jaccard_similarity(center.reshape(-1), neighbor.reshape(-1))
            else:
                raise Exception("distanceType not found")

        # between the center and each neighbors, calculate the cosine similarity of the activations before weight change
        for curr_image in range(num_closePoints + num_backgroundPoints): # 10 = 5 close + 5 background
            pairImg_similarity_before_repChange[0, curr_image] = similarity_between_center_and_neighbors(
                center_before, neighbor_before[curr_image, :], repChange_distanceType_)
            pairImg_similarity_before_coactivation[0, curr_image] = similarity_between_center_and_neighbors(
                center_before, neighbor_before[curr_image, :], coactivation_distanceType_)

        pairImg_similarity_after_repChange = np.zeros((
            num_centerPoints,
            num_closePoints + num_backgroundPoints))
        pairImg_similarity_after_coactivation = np.zeros((
            num_centerPoints,
            num_closePoints + num_backgroundPoints))
        for curr_image in range(num_closePoints + num_backgroundPoints):
            pairImg_similarity_after_repChange[0, curr_image] = similarity_between_center_and_neighbors(
                center_after, neighbor_after[curr_image, :], repChange_distanceType_)
            pairImg_similarity_after_coactivation[0, curr_image] = similarity_between_center_and_neighbors(
                center_after, neighbor_after[curr_image, :], coactivation_distanceType_)

        # for each pair of images, calculate distance between the activations before and after weight change
        representationalChange = pairImg_similarity_after_repChange - pairImg_similarity_before_repChange

        # prepare the data for NMPH
        if co_activationType == 'before':
            co_activations_flatten = pairImg_similarity_before_coactivation.reshape(-1)
        elif co_activationType == 'after':
            co_activations_flatten = pairImg_similarity_after_coactivation.reshape(-1)
        else:
            raise Exception("co_activationType not found")
        representationChange_flatten = representationalChange.reshape(-1)

        return co_activations_flatten, representationChange_flatten

    co_activations_flatten__ = []  # shape = [pair#, batch#]
    representationChange_flatten__ = []  # shape = [pair#, batch#]

    co_activations_flatten__close = []  # shape = [pair#, batch#]
    representationChange_flatten__close = []  # shape = [pair#, batch#]
    co_activations_flatten__background = []  # shape = [pair#, batch#]
    representationChange_flatten__background = []  # shape = [pair#, batch#]

    assert len(layer_b_activations_before) == total_epochs*batch_num_per_epoch
    if testMode:
        BatchNum = testBatchNum
    else:
        BatchNum = len(layer_b_activations_before)
    for curr_batch in tqdm(range(BatchNum)):
        co_activations_flatten_, representationChange_flatten_ = prepare_data_for_NMPH_only_between_centerPoint(
            curr_batch_=curr_batch,
            layer_activations_before=layer_b_activations_before,
            layer_activations_after=layer_b_activations_after,
            repChange_distanceType_=repChange_distanceType,
            coactivation_distanceType_=coactivation_distanceType
        )

        co_activations_flatten__.append(co_activations_flatten_)
        representationChange_flatten__.append(representationChange_flatten_)

        co_activations_flatten__close.append(co_activations_flatten_[:num_closePoints])
        representationChange_flatten__close.append(representationChange_flatten_[:num_closePoints])
        co_activations_flatten__background.append(co_activations_flatten_[num_closePoints:])
        representationChange_flatten__background.append(representationChange_flatten_[num_closePoints:])

    def separate_integration_differentiation(representationChange_flatten_close,
                                             representationChange_flatten_background, title=""):
        def plot_double_bar(values, errors=None, labels=None, title=None, x_label="", y_label="", fontsize=12,
                            set_background_color=False, save_path=None, show_figure=True):
            """
            Plot two bars side by side with optional errors.

            Parameters:
            - values: List of two values for the bars.
            - errors: List of two error values (optional).
            - labels: List of labels for the bars.
            - title: Plot title.
            - x_label: Label for the x-axis.
            - y_label: Label for the y-axis.
            - fontsize: Font size for labels and ticks.
            - set_background_color: Set background color to light gray.
            - save_path: File path for saving the plot (optional).
            - show_figure: Whether to display the plot (default is True).

            Returns:
            None
            """
            fig, ax = plt.subplots(figsize=(fontsize, fontsize / 2))

            # If errors are provided, plot with error bars
            if errors is not None:
                ax.bar([0, 1], values, yerr=[errors[0], errors[1]], align='center', alpha=0.5, ecolor='black',
                       capsize=10)
            else:
                ax.bar([0, 1], values)

            if set_background_color:
                ax.set_facecolor((242 / 256, 242 / 256, 242 / 256))

            ax.set_ylabel(y_label, fontsize=fontsize)
            ax.set_xlabel(x_label, fontsize=fontsize)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(labels, fontsize=fontsize, ha='center')
            ax.set_facecolor((242 / 256, 242 / 256, 242 / 256))

            ax.tick_params(axis='y', labelsize=fontsize)
            ax.set_title(title, fontsize=fontsize)
            ax.yaxis.grid(True)

            _ = plt.tight_layout()

            if save_path is not None:
                plt.savefig(save_path)

            if show_figure:
                _ = plt.show()
            else:
                _ = plt.close()

        representationChange_close = np.asarray(representationChange_flatten_close).reshape(-1)
        representationChange_background = np.asarray(representationChange_flatten_background).reshape(-1)

        mean_close, p5_close, p95_close = cal_resample(representationChange_close, times=5000)
        mean_background, p5_background, p95_background = cal_resample(representationChange_background, times=5000)
        def convert_nan_to_0(temp):
            if np.isnan(temp):
                return 0
            elif temp is None:
                return 0
            else:
                return temp
        mean_close = convert_nan_to_0(mean_close)
        p5_close = convert_nan_to_0(p5_close)
        p95_close = convert_nan_to_0(p95_close)

        mean_background = convert_nan_to_0(mean_background)
        p5_background = convert_nan_to_0(p5_background)
        p95_background = convert_nan_to_0(p95_background)

        plot_double_bar(
            values=[mean_close, mean_background],
            errors=[[mean_close - p5_close, mean_background - p5_background],     # lower
                    [p95_close - mean_close, p95_background - mean_background]],  # higher
            labels=["Close Neighbors", "Background Neighbors"],
            title=f"{title} mean_background/mean_close={mean_background/mean_close:.2f}",
            x_label="",
            y_label="integration score",
            fontsize=12,
            set_background_color=False,
            save_path=None,
            show_figure=True
        )

    separate_integration_differentiation(representationChange_flatten__close, representationChange_flatten__background,
                                         title="Comparison")

    def run_NMPH(co_activations_flatten, rep_changes_flatten, rows=1, cols=1, plotFig=False):
        if plotFig:
            if rows is None:
                rows = int(np.ceil(np.sqrt(len(co_activations_flatten))))
            if cols is None:
                cols = int(np.sqrt(len(co_activations_flatten)))

            fig, axs = plt.subplots(rows, cols, figsize=(15, 15))  # Create a subplot matrix
            cmap = get_cmap('viridis')  # Choose a colormap (you can change 'viridis' to your preferred one)
        else:
            fig = None
            axs = None
            cmap = None

        mean_correlation_coefficients = []
        # recorded_data = []  # Store recorded data for visualization
        mean_parameters = []
        x_partials = []
        y_partials = []
        # for i in tqdm(range(len(co_activations_flatten))):
        x__ = np.asarray(co_activations_flatten).reshape(-1)
        y__ = np.asarray(rep_changes_flatten).reshape(-1)
        labels = np.zeros(np.asarray(co_activations_flatten).shape)
        labels[:,:num_closePoints] = 1  # 1 means close neighbors
        labels[:,num_closePoints:] = 0  # 0 means background neighbors

        mean_correlation_coefficient, mean_parameter, x_partial, y_partial = cubic_fit_correlation_with_params(
            x__, y__,
            n_splits=3,
            random_state=42,
            return_subset=True
        )
        mean_correlation_coefficients.append(mean_correlation_coefficient)
        mean_parameters.append(mean_parameter)
        x_partials.append(x_partial)
        y_partials.append(y_partial)

        if plotFig:
            row = 0
            col = 0

            ax = axs  #[row, col]  # Select the appropriate subplot

            # Color the dots based on a sequence
            sequence = np.linspace(0, 1, len(x__))  # Create a sequence of values from 0 to 1
            colors = cmap(sequence)  # Map the sequence to colors using the chosen colormap

            # ax.scatter(x__, y__, s=10, c=colors)  # 's' controls the size of the points, 'c' sets the colors
            ax.scatter(x__, y__, s=10, c=labels,
                       alpha=0.5
                       )  # 's' controls the size of the points, 'c' sets the colors


            # test whether x__ and y__ are significantly linear regression
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(x__, y__)
            print(f"p value = {p_value}; r_value = {r_value}; slope = {slope}; intercept = {intercept}")


            ax.set_title(f"1 means close nei, 0 means background nei; linear regression p value={p_value:.2f}; r_value={r_value:.2f}")


            # Create a ScalarMappable for the colorbar
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
            sm.set_array([])  # Set an empty array to the ScalarMappable

            colorbar = fig.colorbar(sm, ax=ax)
            # colorbar = fig.colorbar(colors, ax=ax)

        mean_correlation_coefficients = np.array(mean_correlation_coefficients)
        p_value = np.nanmean(mean_correlation_coefficients < 0)
        print(f"p value = {p_value}")

        if plotFig:
            fig, axs = plt.subplots(1, 1, figsize=(15, 15))  # Create a subplot matrix
            cmap = get_cmap('viridis')  # Choose a colormap (you can change 'viridis' to your preferred one)
            # for i in tqdm(range(len(co_activations_flatten))):

            ax = axs  # [row, col]  # Select the appropriate subplot

            # x__ = co_activations_flatten[i]
            # y__ = rep_changes_flatten[i]
            # ax.hist2d(x__, y__, bins=100, cmap=cmap)
            # ax.hist(x__, bins=100, color='blue', alpha=0.5)
            ax.hist(y__, bins=500, color='blue', alpha=0.5)
            # set ylim
            ax.set_ylim([0, 400])
            ax.set_title(f"histogram of representational change aka y axis")

        # Return mean_correlation_coefficients along with recorded_data
        return mean_correlation_coefficients, np.array(mean_parameters), np.array(x_partials), np.array(y_partials)

    if testMode:
        mean_correlation_coefficients_, mean_parameters_, x_partials_, y_partials_ = run_NMPH(
            co_activations_flatten__[:BatchNum - 1], representationChange_flatten__[:BatchNum - 1],
            plotFig=True)
    else:
        mean_correlation_coefficients_, mean_parameters_, x_partials_, y_partials_ = run_NMPH(
            co_activations_flatten__, representationChange_flatten__,
            plotFig=True)


representational_level(
    total_epochs=total_epochs,
    batch_num_per_epoch=1000*num_iterations_per_batch,  # batch_num_per_epoch * iter_per_batch = 1000*3
    num_closePoints=num_close,
    num_backgroundPoints=num_background,
)


def synaptic_level(total_epochs=50, batch_num_per_epoch=1000):
    """

    design the input of NMPH_synaptic should be
        A layer ; before training ; center points
        A layer ; before training ; close neighbors
        A layer ; before training ; background neighbors
        B layer ; before training ; center points
        B layer ; before training ; close neighbors
        B layer ; before training ; background neighbors
        weight change

    """
    import os
    import random
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr
    from scipy.optimize import curve_fit
    from tqdm import tqdm

    test_mode = True
    directory_path = "/gpfs/milgram/scratch60/turk-browne/kp578/LocalAgg/toyData/synaptic_level/"
    os.makedirs(directory_path, exist_ok=True)

    def prepare_data():
        # Set seed
        random.seed(131)

        # Randomly select channel IDs for layers A and B
        unit_num_layer_a = hidden_dim
        unit_num_layer_b = 2
        selected_channel_ids_layer_a = random.sample(range(0, unit_num_layer_a), unit_num_layer_a)
        selected_channel_ids_layer_b = random.sample(range(0, unit_num_layer_b), unit_num_layer_b)

        # Sort the selected channel IDs
        selected_channel_ids_layer_a.sort()
        selected_channel_ids_layer_b.sort()

        # Define paths for data folders
        weight_difference_folder = "/gpfs/milgram/scratch60/turk-browne/kp578/LocalAgg/toyData/weight_difference_folder/"

        # Load data
        weight_difference_history_input_layer = np.load(
            f'{weight_difference_folder}/weight_difference_history_input_layer.npy')
        total_batch_num = weight_difference_history_input_layer.shape[0]
        print(f"Total Batch Num: {total_batch_num}")  # 10000

        # load
        #         A layer ; before training ; center points
        #         A layer ; before training ; close neighbors
        #         A layer ; before training ; background neighbors
        #         B layer ; before training ; center points
        #         B layer ; before training ; close neighbors
        #         B layer ; before training ; background neighbors
        A_layer_before_training_center_points = np.load(
            f'{weight_difference_folder}/A_layer_before_training_center_points.npy')  # shape = (50000, 32)
        # Reshape A_layer_before_training_center_points to (50000, 1, 1, 32)
        A_layer_before_training_center_points = A_layer_before_training_center_points.reshape((
            total_epochs*batch_num_per_epoch, 1, 1, unit_num_layer_a))
        A_layer_before_training_close_neighbors = np.load(
                        f'{weight_difference_folder}/A_layer_before_training_close_neighbors.npy')  # shape = (50000, 1, 5, 32)
        A_layer_before_training_background_neighbors = np.load(
                        f'{weight_difference_folder}/A_layer_before_training_background_neighbors.npy')  # shape = (50000, 1, 5, 32)
        A_layer_before_training_background_neighbors = A_layer_before_training_background_neighbors.reshape((
            total_epochs*batch_num_per_epoch, 1, num_background, unit_num_layer_a))
        layer_a_activations = np.concatenate([
                A_layer_before_training_close_neighbors,
                A_layer_before_training_background_neighbors,
                A_layer_before_training_center_points
            ], axis=2)  # shape = (50000, 1, 11, 32)
        layer_a_activations = layer_a_activations.reshape((
            total_epochs*batch_num_per_epoch, 1+num_close+num_background, unit_num_layer_a))  # shape = (50000, 11, 32)

        B_layer_before_training_center_points = np.load(
            f'{weight_difference_folder}/B_layer_before_training_center_points.npy')
        B_layer_before_training_center_points = B_layer_before_training_center_points.reshape((
            total_epochs*batch_num_per_epoch, 1, 1, unit_num_layer_b))
        B_layer_before_training_close_neighbors =  np.load(
                        f'{weight_difference_folder}/B_layer_before_training_close_neighbors.npy')
        B_layer_before_training_background_neighbors = np.load(
                        f'{weight_difference_folder}/B_layer_before_training_background_neighbors.npy')
        B_layer_before_training_background_neighbors = B_layer_before_training_background_neighbors.reshape((
            total_epochs*batch_num_per_epoch, 1, num_background, unit_num_layer_b))
        layer_b_activations = np.concatenate([
                B_layer_before_training_close_neighbors,
                B_layer_before_training_background_neighbors,
                B_layer_before_training_center_points
            ], axis=2)
        layer_b_activations = layer_b_activations.reshape((
            total_epochs*batch_num_per_epoch, 1+num_close+num_background, unit_num_layer_b))

        weight_changes = np.load(
            f'{weight_difference_folder}/weight_difference_history_hidden_layer2.npy')  # (10000, 2, 32)

        # Obtain co-activations and weight changes
        co_activations_flatten = []
        weight_changes_flatten = []
        pair_ids = []

        for curr_channel_a_feature in tqdm(range(len(selected_channel_ids_layer_a))):  # 32*2 = 64 pairs
            for curr_channel_b_feature in range(len(selected_channel_ids_layer_b)):
                # Extract activations and weight changes for the current channel pair
                activation_layer_a = layer_a_activations[:, :, curr_channel_a_feature]  # (50000, 11, 1)
                activation_layer_b = layer_b_activations[:, :, curr_channel_b_feature]  # (50000, 11, 1)
                weight_change = weight_changes[:, curr_channel_b_feature, curr_channel_a_feature]  # (10000,)

                weight_changes_flatten.append(weight_change)

                # Calculate co-activation
                co_activation = np.multiply(activation_layer_a, activation_layer_b)

                # Average co-activation across the batch
                co_activation = np.mean(co_activation, axis=1)  # (10000,)

                co_activations_flatten.append(co_activation)
                pair_ids.append([
                    selected_channel_ids_layer_a[curr_channel_a_feature],
                    selected_channel_ids_layer_b[curr_channel_b_feature]
                ])

        return co_activations_flatten, weight_changes_flatten, pair_ids

    co_activations_flatten_, weight_changes_flatten_, pair_ids_ = prepare_data()  # co_activations_flatten_ (64, 50000) weight_changes_flatten_ (64, 50000) pair_ids_ (64, 2)

    if not os.path.exists(f'{directory_path}/temp'):
        os.mkdir(f'{directory_path}/temp')

    if not test_mode:
        np.save(f'{directory_path}/temp/co_activations_flatten_.npy',
                co_activations_flatten_)  # shape = [pair#, batch#]
        np.save(f'{directory_path}/temp/weight_changes_flatten_.npy',
                weight_changes_flatten_)  # shape = [pair#, batch#]
        np.save(f'{directory_path}/temp/pair_ids_.npy',
                pair_ids_)  # shape = [pair#, [ID1, ID2]]

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

    def run_NMPH(co_activations_flatten, weight_changes_flatten, pair_ids, rows=None, cols=None, plot_fig=False):
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
        for i in tqdm(range(len(co_activations_flatten))):
            if test_mode:
                test_batch_num = 5000
                x__ = co_activations_flatten[i][:test_batch_num]
                y__ = weight_changes_flatten[i][:test_batch_num]
                pair_id = pair_ids[i]
            else:
                x__ = co_activations_flatten[i]
                y__ = weight_changes_flatten[i]
                pair_id = pair_ids[i]
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
                row = i // cols
                col = i % cols

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

    if test_mode:
        mean_correlation_coefficients_, mean_parameters_, x_partials_, y_partials_ = run_NMPH(
            co_activations_flatten_[:9], weight_changes_flatten_[:9], pair_ids_[:9], plot_fig=True)
    else:
        mean_correlation_coefficients_, mean_parameters_, x_partials_, y_partials_ = run_NMPH(
            co_activations_flatten_, weight_changes_flatten_, pair_ids_)

    if not test_mode:
        np.save(f'{directory_path}/temp/mean_correlation_coefficients_.npy', mean_correlation_coefficients_)
        np.save(f'{directory_path}/temp/mean_parameters_.npy', mean_parameters_)
        np.save(f'{directory_path}/temp/x_partials_.npy', x_partials_)
        np.save(f'{directory_path}/temp/y_partials_.npy', y_partials_)

    x_partials_ = x_partials_.flatten()
    y_partials_ = y_partials_.flatten()
    mean_parameters_avg = np.mean(mean_parameters_, axis=0)

    def plot_scatter_and_cubic(x_partials, y_partials, mean_parameters):
        def cubic_function(_x, a, b, c, d):
            print(f"a={a}, b={b}, c={c}, d={d}")
            return a * _x ** 3 + b * _x ** 2 + c * _x + d

        # Scatter plot
        plt.scatter(x_partials, y_partials, label='Data Points', color='green', marker='o', s=30)

        # Fit cubic curve using curve_fit
        # popt, _ = curve_fit(cubic_function, x_partials_, y_partials_)

        # Generate points for the fitted cubic curve
        x_fit = np.linspace(min(x_partials), max(x_partials), 100)
        y_fit = cubic_function(x_fit, *mean_parameters)

        # Plot the fitted cubic curve
        plt.plot(x_fit, y_fit, label='Fitted Cubic Curve', color='red')

        # Add labels and a legend
        plt.xlabel('X Partials')
        plt.ylabel('Y Partials')
        plt.legend()

        # Show the plot
        plt.show()

    # plot_scatter_and_cubic(x_partials_, y_partials_, mean_parameters_avg)


synaptic_level(
    total_epochs=total_epochs,
    batch_num_per_epoch=1000*num_iterations_per_batch  # batch_num_per_epoch * iter_per_batch = 1000*3)
    )

print('done')


def NMPH_lineFit():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    # Define the NMPHcurve function
    def NMPHcurve(x, DThr=None, DRevMag=None, DRev=None, ThrP=None):
        return np.piecewise(x,
                            [x < DThr, (x >= DThr) & (x < DRev), (x >= DRev) & (x <= 1)],
                            [lambda x: 0,
                             lambda x: (DRevMag - 0.0) / (DRev - DThr) * (x - DThr),
                             lambda x: (0.0 - DRevMag) / (ThrP - DRev) * (x - ThrP)])

    # Set parameters
    DThr = 0.33
    DRevMag = -0.25
    DRev = 0.67
    ThrP = 0.76
    DMaxMag = (0.0 - DRevMag) / (ThrP - DRev) * (1 - ThrP)

    print(
        f"real parameters: \nDThr={DThr:.2f}, DRevMag={DRevMag:.2f}, DRev={DRev:.2f}, ThrP={ThrP:.2f}, DMaxMag={DMaxMag:.2f}")

    # Generate x values
    x_values = np.linspace(0, 1, 100)

    # Calculate corresponding y values using NMPHcurve
    y_values = NMPHcurve(x_values, DThr, DRevMag, DRev, ThrP)

    # Add jittering to y values
    jitter = np.random.normal(0, 0.05, len(y_values))
    y_values_jittered = y_values + jitter
    x_data = x_values
    y_data = y_values_jittered

    # Define the bounds for parameters
    lower_bounds = [0, -1, 0, 0]  # Adjusted lower bounds for DRevMag
    upper_bounds = [1, 0, 1, 1]

    # Use curve_fit to fit NMPHcurve to the generated data
    params, covariance = curve_fit(NMPHcurve, x_data, y_data,
                                   bounds=(lower_bounds, upper_bounds),
                                   p0=[0.1, -0.5, 0.5, 0.8]   # DThr=None, DRevMag=None, DRev=None, ThrP=None
                                   )

    # Print the fitted parameters
    print("Fitted Parameters:", end='\n')
    print(f"DThr={params[0]:.2f}", end=', ')
    print(f"DRevMag={params[1]:.2f}", end=', ')
    print(f"DRev={params[2]:.2f}", end=', ')
    print(f"ThrP={params[3]:.2f}", end=', ')
    DMaxMag = (0.0 - params[1]) / (params[3] - params[2]) * (1 - params[3])
    print(f"DMaxMag={DMaxMag:.2f}")

    # Generate x values for plotting
    x_fit = np.linspace(0, 1, 100)

    # Calculate y values using the fitted parameters
    y_fit = NMPHcurve(x_fit, *params)

    # Plot the original data points and the fitted curve
    plt.scatter(x_data, y_data, label='Random Data Points', marker='o')
    plt.plot(x_fit, y_fit, label='Fitted NMPH Curve', color='red')

    # Add labels and title
    plt.title('Fitting NMPH Curve to Random Data Points')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # Show legend
    plt.legend()

    # Show the plot
    plt.show()
