import numpy as np

# Domain settings (2D rod)
x_min, x_max = 0.0, 10.0  # Length of the rod
y_min, y_max = 0.0, 2.0   # Width of the rod
t_min, t_max = 0.0, 50.0  # Time range

# Thermal properties
alpha = 0.1  # Thermal diffusivity (m²/s)

# Boundary and initial conditions
T_left = 100.0   # Temperature at left boundary (°C)
T_initial = 10.0 # Initial temperature (°C)

# Neural network settings
layers = [3, 64, 64, 64, 64, 1]  # Input: (x, y, t), Output: T

# Training settings
n_domain = 15000   # Number of points inside domain
n_boundary = 4000  # Number of points on boundaries
n_initial = 3000   # Number of points at initial time
epochs = 15000
batch_size = 1024
learning_rate = 0.001

# Sampling parameters
sample_every = 1000  # Save model every N epochs
plot_every = 2000    # Create visualization every N epochs

# Path settings
model_save_path = "./models"
results_path = "./results"
