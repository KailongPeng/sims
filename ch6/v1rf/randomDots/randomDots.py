import numpy as np
import matplotlib.pyplot as plt

n = 21
lambda_factor = 0.1  # Example scaling factor
iterations = 1000  # Number of iterations for optimization

# Generate n points uniformly distributed in a 2D plane [0, 1] x [0, 1]
points = np.random.rand(n, 2)


# Function to calculate the distance matrix
def calculate_distance_matrix(points):
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i, j] = np.linalg.norm(points[i] - points[j])
    return dist_matrix


# Calculate initial distance matrix
initial_dist_matrix = calculate_distance_matrix(points)


# Function to move points randomly controlled by lambda
def move_points_randomly(points, lambda_factor):
    new_points = points + (np.random.rand(n, 2) - 0.5) * lambda_factor
    return new_points


# Function to calculate objective and scatter plot data
def calculate_objective_and_plot_data(initial_dist_matrix, new_dist_matrix):
    m1 = initial_dist_matrix[np.triu_indices(n, k=1)]
    m2 = new_dist_matrix[np.triu_indices(n, k=1)]

    x = -m1
    y = m2 - m1

    # Sort by x values
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]

    third = len(x_sorted) // 3
    noChange = y_sorted[:third]
    differentiation = y_sorted[third:2 * third]
    integration = y_sorted[2 * third:]

    obj_noChange = np.sum(np.abs(noChange))
    obj_differentiation = np.sum(differentiation)
    obj_integration = -np.sum(integration)

    objective = obj_noChange + obj_differentiation + obj_integration
    return objective, x, y


# Initial scatter plot data
initial_objective, initial_x, initial_y = calculate_objective_and_plot_data(initial_dist_matrix, initial_dist_matrix)

# Optimize the positions
best_points = points.copy()
best_objective = float('inf')
best_x = None
best_y = None

for _ in range(iterations):
    new_points = move_points_randomly(points, lambda_factor)
    new_dist_matrix = calculate_distance_matrix(new_points)
    new_objective, new_x, new_y = calculate_objective_and_plot_data(initial_dist_matrix, new_dist_matrix)

    if new_objective < best_objective:
        best_points = new_points.copy()
        best_objective = new_objective
        best_x = new_x
        best_y = new_y

# Plot initial and final positions
plt.figure(figsize=(12, 12))

plt.subplot(2, 2, 1)
plt.scatter(points[:, 0], points[:, 1], c='blue', label='Initial')
plt.title("Initial Positions")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()

plt.subplot(2, 2, 2)
plt.scatter(best_points[:, 0], best_points[:, 1], c='red', label='Final')
plt.title("Final Positions")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()

# Plot initial and final scatter plot of objective
plt.subplot(2, 2, 3)
plt.scatter(initial_x, initial_y, c='blue', label='Initial')
plt.title("Initial Objective Scatter Plot")
plt.xlabel("-m1")
plt.ylabel("m2 - m1")
plt.legend()

plt.subplot(2, 2, 4)
plt.scatter(best_x, best_y, c='red', label='Final')
plt.title("Final Objective Scatter Plot")
plt.xlabel("-m1")
plt.ylabel("m2 - m1")
plt.legend()

plt.tight_layout()
plt.show()
