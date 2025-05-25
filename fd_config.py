import numpy as np

# Domain settings (2D rod)
x_min, x_max = 0.0, 10.0  # Length of the rod (m)
y_min, y_max = 0.0, 2.0   # Width of the rod (m)
t_min, t_max = 0.0, 50.0  # Time range (s)

# Grid resolution
nx = 400  # Number of grid points in x direction
ny = 80   # Number of grid points in y direction
nt = 2000 # Number of time steps

# Calculate grid spacing
dx = (x_max - x_min) / (nx - 1)
dy = (y_max - y_min) / (ny - 1)
dt = (t_max - t_min) / nt

# Thermal properties
alpha = 0.1  # Thermal diffusivity (m²/s)

# Boundary and initial conditions
T_left = 100.0   # Temperature at left boundary (°C)
T_initial = 10.0 # Initial temperature (°C)

# Stability condition for explicit finite difference
# For 2D: dt <= dx²dy²/(2α(dx² + dy²))
max_dt = (dx**2 * dy**2) / (2 * alpha * (dx**2 + dy**2))
print(f"Maximum stable dt: {max_dt:.6f} s")
print(f"Current dt: {dt:.6f} s")

if dt > max_dt:
    print("WARNING: Current dt exceeds stability limit!")
    # Adjust dt to be stable
    dt = 0.8 * max_dt
    nt = int((t_max - t_min) / dt)
    print(f"Adjusted dt: {dt:.6f} s")
    print(f"Adjusted nt: {nt}")

# Finite difference coefficients
r_x = alpha * dt / dx**2
r_y = alpha * dt / dy**2

print(f"Stability parameters: r_x = {r_x:.4f}, r_y = {r_y:.4f}")
print(f"Sum r_x + r_y = {r_x + r_y:.4f} (should be ≤ 0.5)")

# Animation settings
animation_frames = 100
frame_interval = max(1, nt // animation_frames)

# Path settings
results_path = "./results/finite_difference"
