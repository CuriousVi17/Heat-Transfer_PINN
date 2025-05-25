import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
from matplotlib.colors import Normalize
import argparse
from model import HeatPINN
from utils import plot_temperature_field
from config import (
    x_min, x_max, y_min, y_max, alpha, T_left, T_initial
)
import warnings
from matplotlib import MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

def generate_grid(nx=400, ny=80):
    """Generate a grid for visualization."""
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x, y)
    
    x_flat = X.flatten()
    y_flat = Y.flatten()
    
    return x_flat, y_flat, X, Y

def get_temperature_range(model, t_values):
    """Calculate the global temperature range across all time steps for fixed colormap."""
    print("Calculating global temperature range for fixed colormap...")
    
    nx, ny = 400, 80
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x, y)
    x_flat = X.flatten()
    y_flat = Y.flatten()
    
    global_min = float('inf')
    global_max = float('-inf')
    
    # Sample every few time points to get a good estimate
    sample_indices = np.linspace(0, len(t_values)-1, min(20, len(t_values)), dtype=int)
    
    for i in sample_indices:
        t = t_values[i]
        t_flat = np.ones_like(x_flat) * t
        
        points = tf.convert_to_tensor(np.stack([x_flat, y_flat, t_flat], axis=1), dtype=tf.float64)
        temperature = model(points).numpy().flatten()
        
        current_min = np.min(temperature)
        current_max = np.max(temperature)
        
        global_min = min(global_min, current_min)
        global_max = max(global_max, current_max)
    
    # Add small buffer to ensure all values are covered
    buffer = (global_max - global_min) * 0.05
    global_min -= buffer
    global_max += buffer
    
    print(f"Global temperature range: {global_min:.2f} - {global_max:.2f}")
    return global_min, global_max

def create_temperature_animation(model, t_values, filename='temperature_animation.gif'):
    """Create an animation of the temperature field over time with fixed colormap."""
    
    # Calculate global temperature range for fixed colormap
    temp_min, temp_max = get_temperature_range(model, t_values)
    
    frames = []
    
    print("Generating animation frames...")
    for i, t in enumerate(t_values):
        if i % 10 == 0:
            print(f"Processing frame {i+1}/{len(t_values)}")
            
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create grid
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

        # Plot temperature field with fixed colormap range
        levels = np.linspace(temp_min, temp_max, 50)
        contour = plt.contourf(X, Y, T_grid, levels=levels, cmap='turbo', 
                              vmin=temp_min, vmax=temp_max, extend='both')
        
        # Add colorbar with fixed range
        cbar = plt.colorbar(contour, label='Temperature (°C)')
        cbar.set_ticks(np.linspace(temp_min, temp_max, 6))

        # Add boundary annotations
        x_rect = [x_min, x_max, x_max, x_min, x_min]
        y_rect = [y_min, y_min, y_max, y_max, y_min]

        plt.plot(x_rect, y_rect, color='black', linewidth=2)

        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title(f'Temperature Distribution at t = {t:.2f} s')
        plt.axis('equal')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        plt.tight_layout()
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
        plt.close()

    imageio.mimsave(filename, frames, fps=5)
    print(f"Temperature animation saved to {filename}")

def extract_thermal_features(model, t):
    """Extract and analyze thermal features at time t."""
    
    x_valid, y_valid, X, Y = generate_grid(nx=300, ny=150)
    t_valid = np.ones_like(x_valid) * t

    # Get temperature field
    points = tf.convert_to_tensor(np.stack([x_valid, y_valid, t_valid], axis=1), dtype=tf.float64)
    temperature = model(points).numpy().flatten()

    # Analyze different regions
    # Left region (near heated boundary)
    left_mask = (x_valid < x_min + 1.0) & (y_valid > y_min + 0.2) & (y_valid < y_max - 0.2)
    
    # Center region
    center_mask = (x_valid > 4.0) & (x_valid < 6.0) & (y_valid > y_min + 0.2) & (y_valid < y_max - 0.2)
    
    # Right region (far from heated boundary)
    right_mask = (x_valid > x_max - 1.0) & (y_valid > y_min + 0.2) & (y_valid < y_max - 0.2)

    stats = {
        'max_temperature': np.max(temperature),
        'min_temperature': np.min(temperature),
        'avg_temperature': np.mean(temperature),
        'left_avg_temp': np.mean(temperature[left_mask]) if np.sum(left_mask) > 0 else 0,
        'center_avg_temp': np.mean(temperature[center_mask]) if np.sum(center_mask) > 0 else 0,
        'right_avg_temp': np.mean(temperature[right_mask]) if np.sum(right_mask) > 0 else 0,
    }
    
    return stats

def main():
    parser = argparse.ArgumentParser(description='Simulate and visualize 2D heat diffusion')
    parser.add_argument('--model_path', type=str, default='./models/model_final',
                      help='Path to the trained model')
    parser.add_argument('--output_dir', type=str, default='./results/simulation',
                      help='Directory to save simulation results')
    parser.add_argument('--t_start', type=float, default=0.01,
                      help='Start time for simulation')
    parser.add_argument('--t_end', type=float, default=200.0,
                      help='End time for simulation')
    parser.add_argument('--num_frames', type=int, default=100,
                      help='Number of frames for animation')
    parser.add_argument('--temp_min', type=float, default=None,
                      help='Minimum temperature for colormap (if not specified, calculated automatically)')
    parser.add_argument('--temp_max', type=float, default=None,
                      help='Maximum temperature for colormap (if not specified, calculated automatically)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Load model
    model = HeatPINN()
    
    # Initialize model with dummy input
    dummy_input = tf.zeros((1, 3), dtype=tf.float64)
    _ = model(dummy_input)
    
    # Load trained weights
    model.load_weights(args.model_path)
    print(f"Model loaded from {args.model_path}")
    
    # Generate time values for animation
    t_values = np.linspace(args.t_start, args.t_end, args.num_frames)
    
    # Override temperature range calculation if provided
    if args.temp_min is not None and args.temp_max is not None:
        print(f"Using user-specified temperature range: {args.temp_min} - {args.temp_max}")
        # Monkey patch the range calculation function
        original_get_range = get_temperature_range
        def fixed_get_range(model, t_values):
            return args.temp_min, args.temp_max
        globals()['get_temperature_range'] = fixed_get_range
    
    # Create animations
    print("\nCreating animations...")
    create_temperature_animation(model, t_values, 
                               filename=os.path.join(args.output_dir, 'temperature_animation.gif'))
    
    # Analyze specific time points
    times_to_analyze = [1.0, 5.0, 10.0, 20.0, 30.0, 50.0, 100.0]
    all_stats = {}
    
    print("\n" + "="*50)
    print("THERMAL FEATURE ANALYSIS")
    print("="*50)
    
    for t in times_to_analyze:
        stats = extract_thermal_features(model, t)
        all_stats[t] = stats
        
        print(f"\nThermal statistics at t = {t:.1f} s:")
        for key, value in stats.items():
            print(f"  {key}: {value:.6f}")
    
    # Generate static plots for key time points with fixed colormap
    print("\nGenerating static plots...")
    
    # Calculate temperature range for static plots (reuse from animation if available)
    print("Calculating temperature range for static plots...")
    static_times = [1.0, 5.0, 10.0, 25.0, 50.0, 100.0]
    
    if args.temp_min is not None and args.temp_max is not None:
        temp_range = (args.temp_min, args.temp_max)
    else:
        from utils import get_global_temperature_range
        temp_range = get_global_temperature_range(model, t_values=np.array(static_times))
    
    print(f"Using temperature range for static plots: {temp_range[0]:.2f} - {temp_range[1]:.2f}")
    
    for t in static_times:
        plot_temperature_field(model, t, 
                             save_path=os.path.join(args.output_dir, f'temperature_t{t}.png'),
                             temp_range=temp_range)
    
    # Create summary report
    create_summary_report(all_stats, args.output_dir)
    
    print(f"\nSimulation completed! Results saved to {args.output_dir}")

def create_summary_report(all_stats, output_dir):
    """Create a summary report of the thermal analysis."""
    report_path = os.path.join(output_dir, 'thermal_analysis_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        
        f.write("THERMAL DIFFUSION SIMULATION REPORT\n")
        f.write("="*50 + "\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  Domain: {x_min} ≤ x ≤ {x_max}, {y_min} ≤ y ≤ {y_max}\n")
        f.write(f"  Thermal diffusivity (α): {alpha}\n")
        f.write(f"  Boundary temperature: {T_left}\n")
        f.write(f"  Initial temperature: {T_initial}\n\n")
        
        f.write("Time Evolution Analysis:\n")
        f.write("-" * 30 + "\n")
        
        for t, stats in all_stats.items():
            f.write(f"\nTime t = {t:.1f} s:\n")
            f.write(f"  Temperature range: {stats['min_temperature']:.2f} - {stats['max_temperature']:.2f}\n")
            f.write(f"  Average temperature: {stats['avg_temperature']:.2f}\n")
            
            if stats['left_avg_temp'] > 0:
                f.write(f"  Left region avg temp: {stats['left_avg_temp']:.2f}\n")
            if stats['center_avg_temp'] > 0:
                f.write(f"  Center region avg temp: {stats['center_avg_temp']:.2f}\n")
            if stats['right_avg_temp'] > 0:
                f.write(f"  Right region avg temp: {stats['right_avg_temp']:.2f}\n")
    
    print(f"Summary report saved to {report_path}")

if __name__ == "__main__":
    main()
