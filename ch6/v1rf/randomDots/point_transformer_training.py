import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

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

    # 放弃记录所有的weight changes和activations，只记录每一个curr_timepoint的最后一个epoch完成后的weight of the third layer (fc3: (40, 20) -> (40, 2)) 和 activations of layer 3 (shape: (40, 20) and output layer (shape: (40, 2))


# Early stopping class
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
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
def train_model(best_points_history, patience=10, min_delta=0, max_epochs=10000):
    initial_points = best_points_history[0]  # shape: (40, 2)
    best_points_history = torch.tensor(best_points_history, dtype=torch.float32)
    initial_points = torch.tensor(initial_points, dtype=torch.float32)

    model = PointTransformer()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=0)

    final_weight_changes = []
    final_activations_layer3 = []
    final_activations_output = []

    losses = {}
    for curr_timepoint in range(1, best_points_history.shape[0]):
        early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
        for epoch in tqdm(range(max_epochs)):
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
                break

        # Record the final weight and activations after last epoch of each curr_timepoint
        with torch.no_grad():
            final_weight_changes.append(model.fc3.weight.clone().detach().numpy())
            activation_layer3 = torch.relu(model.fc2(torch.relu(model.fc1(initial_points))))
            final_activations_layer3.append(activation_layer3.numpy())
            final_activations_output.append(outputs.numpy())

    for curr_timepoint in range(1, best_points_history.shape[0]):
        plt.figure(figsize=(10, 5))
        plt.plot(losses[curr_timepoint], label=f'Time Point {curr_timepoint}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Loss Curve for Time Point {curr_timepoint}')
        plt.legend()
        plt.show()

    return model, final_weight_changes, final_activations_layer3, final_activations_output

best_points_history = np.asarray(np.load('./result/best_points_history.npy'))
best_points_history = best_points_history[0:3]
model, final_weight_changes, final_activations_layer3, final_activations_output = train_model(
    best_points_history, patience=100, min_delta=1e-20, max_epochs=10000)

final_weight_changes = np.array(final_weight_changes)
final_activations_layer3 = np.array(final_activations_layer3)
final_activations_output = np.array(final_activations_output)
print(final_weight_changes.shape)
print(final_activations_layer3.shape)
print(final_activations_output.shape)
