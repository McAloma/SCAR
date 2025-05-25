import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

# Generate 100 fixed 2D points
np.random.seed(0)
points = np.random.rand(100, 2)

# Proximity values
proximity_values = points[:, 0] + points[:, 1]
normalized_proximity = MinMaxScaler().fit_transform(proximity_values.reshape(-1, 1)).flatten()

# Case 3a: Simple linear separation based on proximity values (2 categories)
labels_case3a = (proximity_values > np.median(proximity_values)).astype(int)
colors_case3a = plt.cm.Paired(labels_case3a / 2.0)  # Map 0,1 to 2-color map

# Case 3b: More complex linear separation (3 categories)
labels_case3b = np.digitize(proximity_values, bins=[0.66, 1.33])
colors_case3b = plt.cm.Set2(labels_case3b / 2.0)  # Map 0,1,2 to 3-color map

# Create output directory if not exists
output_dir = './draw'
os.makedirs(output_dir, exist_ok=True)

# Function to plot with decision boundary
def plot_with_decision_boundary(points, labels, colors, boundary_condition, file_path, title):
    plt.figure(figsize=(5, 4))
    plt.scatter(points[:, 0], points[:, 1], color=colors)
    
    # Plot decision boundary (based on labels and conditions)
    x_min, x_max = points[:, 0].min() - 0.1, points[:, 0].max() + 0.1
    y_min, y_max = points[:, 1].min() - 0.1, points[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    # Apply boundary condition for decision-making
    Z = boundary_condition(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and fill regions
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)

# Case 3a: Simple linear separation (2 colors) with boundary
file_path_3a = os.path.join(output_dir, "Subtask A with Boundary.png")
boundary_condition_3a = lambda x: (x[:, 0] + x[:, 1]) > np.median(proximity_values)  # Decision boundary condition
plot_with_decision_boundary(points, labels_case3a, colors_case3a, boundary_condition_3a, file_path_3a, "Subtask A with Boundary")

# Case 3b: Complex linear separation (3 colors) with boundary
file_path_3b = os.path.join(output_dir, "Subtask B with Boundary.png")
boundary_condition_3b = lambda x: np.digitize(x[:, 0] + x[:, 1], bins=[0.66, 1.33])  # Decision boundary condition
plot_with_decision_boundary(points, labels_case3b, colors_case3b, boundary_condition_3b, file_path_3b, "Subtask B with Boundary")

# Returning file paths for confirmation
file_paths = [file_path_3a, file_path_3b]
file_paths