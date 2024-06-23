import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


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
def train_model(best_points_history, num_epochs=100):
    initial_points = best_points_history[0]  # shape: (40, 2)
    best_points_history = torch.tensor(best_points_history, dtype=torch.float32)
    initial_points = torch.tensor(initial_points, dtype=torch.float32)

    model = PointTransformer()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Storage for weight changes and activations
    weight_changes = []
    activations = []

    for epoch in range(num_epochs):
        for t in range(1, best_points_history.shape[0]):
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

    return model, weight_changes, activations


# Example usage
best_points_history = np.random.rand(101, 40, 2)  # Replace with actual data
model, weight_changes, activations = train_model(best_points_history)

# Now `weight_changes` and `activations` hold the required information
