import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

import os

if 'gpfs/milgram' in os.getcwd():
    os.chdir('"/gpfs/milgram/project/turk-browne/projects/sandbox/simulation/sims/ch6/v1rf/randomDots"')
else:
    os.chdir(r"D:\Desktop\simulation\sims\ch6\v1rf\randomDots")

print(f"Current working directory: {os.getcwd()}")
# torch version
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


# Function to train the model and record weights and activations
def train_model(best_points_history, num_epochs=10):
    initial_points = best_points_history[0]  # shape: (40, 2)
    best_points_history = torch.tensor(best_points_history, dtype=torch.float32)
    initial_points = torch.tensor(initial_points, dtype=torch.float32)

    model = PointTransformer()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Storage for weight changes and activations
    weight_changes = []
    activations = []

    for t in range(1, best_points_history.shape[0]):
        for epoch in tqdm(range(num_epochs)):
            optimizer.zero_grad()
            outputs = model(initial_points)
            target = best_points_history[t]
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            # Record the weight changes of the third layer (fc3)
            weight_change = model.fc3.weight.grad.clone().detach().numpy()
            weight_changes.append(weight_change)

            # Record the activations of the third layer before the output layer
            with torch.no_grad():
                activation = torch.relu(model.fc2(torch.relu(model.fc1(initial_points)))).numpy()
            activations.append(activation)

            # Visualize training outputs and targets
            if epoch % (num_epochs // 5) == 0:  # Plot every few epochs
                plt.figure(figsize=(10, 5))
                plt.scatter(outputs[:, 0].detach().numpy(), outputs[:, 1].detach().numpy(), color='blue', label='Predicted Points')
                plt.scatter(target[:, 0].detach().numpy(), target[:, 1].detach().numpy(), color='red', label='Target Points')
                plt.title(f'Time Point {t}, Epoch {epoch}')
                plt.legend()
                plt.show()

    return model, weight_changes, activations


best_points_history = np.asarray(np.load('./result/best_points_history.npy'))  # (np.array(best_points_history).shape=(101, 40, 2))。
best_points_history = best_points_history[0:3]
model, weight_changes, activations = train_model(best_points_history, num_epochs=10)

weight_changes = np.array(weight_changes)
activations = np.array(activations)
print(weight_changes.shape)  # (20, 2, 20)
print(activations.shape)  # (20, 40, 20)
