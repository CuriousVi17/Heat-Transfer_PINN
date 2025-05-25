import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from matplotlib.colors import Normalize
from config import x_min, x_max, y_min, y_max

def generate_domain_points(n_points, t_min, t_max):
    """Generate random points in the domain."""
    x = np.random.uniform(x_min, x_max, n_points)
    y = np.random.uniform(y_min, y_max, n_points)
    t = np.random.uniform(t_min, t_max, n_points)
    
    return np.stack([x, y, t], axis=1)

def generate_boundary_points(n_points, t_min, t_max):
    """Generate points on the domain boundaries."""
    n_per_edge = n_points // 4
    t_vals = np.random.uniform(t_min, t_max, n_points)
    
    # Left boundary (heated edge)
    x_left = np.ones(n_per_edge) * x_min
    y_left = np.random.uniform(y_min, y_max, n_per_edge)
    
    # Right boundary (insulated)
    x_right = np.ones(n_per_edge) * x_max
    y_right = np.random.uniform(y_min, y_max, n_per_edge)
    
    # Bottom boundary (insulated)
    x_bottom = np.random.uniform(x_min, x_max, n_per_edge)
    y_bottom = np.ones(n_per_edge) * y_min
    
    # Top boundary (insulated)
    x_top = np.random.uniform(x_min, x_max, n_per_edge)
    y_top = np.ones(n_per_edge) * y_max
    
    # Combine all boundaries
    x = np.concatenate([x_left, x_right, x_bottom, x_top])
    y = np.concatenate([y_left, y_right, y_bottom, y_top])
    t = t_vals[:len(x)]
    
    # Create indicator for boundary type (0: left, 1: right, 2: bottom, 3: top)
    boundary_type = np.concatenate([
        np.zeros(n_per_edge),
        np.ones(n_per_edge),
        np.ones(n_per_edge) * 2,
        np.ones(n_per_edge) * 3
    ])
    
    return np.stack([x, y, t], axis=1), boundary_type

def generate_initial_points(n_points):
    """Generate points at initial time t=0."""
    x = np.random.uniform(x_min, x_max, n_points)
    y = np.random.uniform(y_min, y_max, n_points)
    t = np.zeros(n_points)  # t=0
    
    return np.stack([x, y, t], axis=1)

def get_global_temperature_range(model, t_values=None, t_min=0.01, t_max=200.0, n_samples=20):
    """Calculate global temperature range for consistent colormap across all plots."""
    if t_values is None:
        t_values = np.linspace(t_min, t_max, n_samples)
    elif len(t_values) > n_samples:
        # Sample from provided t_values if too many
        indices = np.linspace(0, len(t_values)-1, n_samples, dtype=int)
        t_values = t_values[indices]
    
    # Use coarser grid for range calculation
    nx, ny = 100, 50
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x, y)
    x_flat = X.flatten()
    y_flat = Y.flatten()
    
    global_min = float('inf')
    global_max = float('-inf')
    
    for t in t_values:
        t_flat = np.ones_like(x_flat) * t
        
        points = tf.convert_to_tensor(np.stack([x_flat, y_flat, t_flat], axis=1), dtype=tf.float64)
        temperature = model(points).numpy().flatten()
        
        current_min = np.min(temperature)
        current_max = np.max(temperature)
        
        global_min = min(global_min, current_min)
        global_max = max(global_max, current_max)
    
    # Add small buffer
    buffer = (global_max - global_min) * 0.05
    global_min -= buffer
    global_max += buffer
    
    return global_min, global_max

def plot_temperature_field(model, t, save_path=None, temp_range=None, **kwargs):
    """
    Plot the temperature field at a specific time t.
    
    Parameters:
    -----------
    model : HeatPINN
        The trained model
    t : float
        Time point
    save_path : str, optional
        Path to save the plot
    temp_range : tuple, optional
        (min_temp, max_temp) for fixed colormap. If None, uses adaptive colormap.
    **kwargs : dict
        Additional arguments:
        - figsize : tuple, figure size (default: (12, 6))
        - dpi : int, resolution for saved plots (default: 300)
        - cmap : str, colormap name (default: 'turbo')
        - levels : int, number of contour levels (default: 50)
    """
    # Extract kwargs with defaults
    figsize = kwargs.get('figsize', (12, 6))
    dpi = kwargs.get('dpi', 300)
    cmap = kwargs.get('cmap', 'turbo')
    levels_count = kwargs.get('levels', 50)
    
    # Create a grid
    nx, ny = 150, 50
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x, y)
    
    # Prepare inputs
    x_flat = X.flatten()
    y_flat = Y.flatten()
    t_flat = np.ones_like(x_flat) * t
    
    # Get temperature predictions
    points = tf.convert_to_tensor(np.stack([x_flat, y_flat, t_flat], axis=1), dtype=tf.float64)
    temperature = model(points).numpy().flatten()
    
    # Reshape to grid
    T_grid = temperature.reshape(ny, nx)
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Plot temperature field with fixed or adaptive colormap
    if temp_range is not None:
        temp_min, temp_max = temp_range
        levels = np.linspace(temp_min, temp_max, levels_count)
        contour = plt.contourf(X, Y, T_grid, levels=levels, cmap=cmap, 
                              vmin=temp_min, vmax=temp_max, extend='both')
        
        # Add colorbar with fixed ticks
        cbar = plt.colorbar(contour, label='Temperature (°C)')
        cbar.set_ticks(np.linspace(temp_min, temp_max, 6))
    else:
        # Adaptive colormap (original behavior)
        contour = plt.contourf(X, Y, T_grid, levels=levels_count, cmap=cmap)
        plt.colorbar(contour, label='Temperature (°C)')
    
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title(f'Temperature Distribution at t = {t:.2f} s')
    plt.axis('equal')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    # Add boundary annotations
    x_rect = [x_min, x_max, x_max, x_min, x_min]
    y_rect = [y_min, y_min, y_max, y_max, y_min]

    plt.plot(x_rect, y_rect, color='black', linewidth=2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_multiple_temperature_fields(model, time_points, save_dir=None, temp_range=None, 
                                   subplot_layout=None, **kwargs):
    """
    Plot multiple temperature fields in a single figure for comparison.
    
    Parameters:
    -----------
    model : HeatPINN
        The trained model
    time_points : list
        List of time points to plot
    save_dir : str, optional
        Directory to save individual plots and comparison plot
    temp_range : tuple, optional
        (min_temp, max_temp) for fixed colormap
    subplot_layout : tuple, optional
        (rows, cols) for subplot layout. If None, determined automatically.
    **kwargs : dict
        Additional plotting arguments
    """
    if temp_range is None:
        print("Calculating global temperature range for consistent comparison...")
        temp_range = get_global_temperature_range(model, t_values=np.array(time_points))
    
    # Determine subplot layout
    n_plots = len(time_points)
    if subplot_layout is None:
        cols = min(4, n_plots)
        rows = (n_plots + cols - 1) // cols
    else:
        rows, cols = subplot_layout
    
    # Create comparison plot
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    if n_plots == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    temp_min, temp_max = temp_range
    levels = np.linspace(temp_min, temp_max, 30)
    
    for i, t in enumerate(time_points):
        ax = axes[i] if i < len(axes) else None
        if ax is None:
            break
            
        # Create grid
        nx, ny = 100, 40  # Smaller grid for subplots
        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        X, Y = np.meshgrid(x, y)
        
        # Get temperature
        x_flat = X.flatten()
        y_flat = Y.flatten()
        t_flat = np.ones_like(x_flat) * t
        
        points = tf.convert_to_tensor(np.stack([x_flat, y_flat, t_flat], axis=1), dtype=tf.float64)
        temperature = model(points).numpy().flatten()
        T_grid = temperature.reshape(ny, nx)
        
        # Plot
        contour = ax.contourf(X, Y, T_grid, levels=levels, cmap='turbo',
                             vmin=temp_min, vmax=temp_max, extend='both')
        
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title(f't = {t:.1f} s')
        ax.set_aspect('equal')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # Add boundary
        x_rect = [x_min, x_max, x_max, x_min, x_min]
        y_rect = [y_min, y_min, y_max, y_max, y_min]
        ax.plot(x_rect, y_rect, color='black', linewidth=1)
    
    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    # Add shared colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(contour, cax=cbar_ax, label='Temperature (°C)')
    cbar.set_ticks(np.linspace(temp_min, temp_max, 6))
    
    plt.tight_layout()
    
    if save_dir:
        comparison_path = os.path.join(save_dir, 'temperature_comparison.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {comparison_path}")
        
        # Save individual plots with consistent colormap
        for t in time_points:
            individual_path = os.path.join(save_dir, f'temperature_t{t:.1f}_fixed.png')
            plot_temperature_field(model, t, save_path=individual_path, 
                                 temp_range=temp_range, **kwargs)
    
    plt.show()

def create_directories():
    """Create necessary directories for saving results."""
    directories = [
        './models', 
        './results', 
        './results/temperature', 
        './results/simulation',
    ]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
