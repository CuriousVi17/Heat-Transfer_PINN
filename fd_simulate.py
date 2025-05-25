import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import imageio
from fd_config import *
from fd_solver import HeatDiffusionSolver
import argparse
import warnings
from matplotlib import MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

class HeatDiffusionSimulator:
    """Simulator for visualizing 2D heat diffusion results."""
    
    def __init__(self, solver):
        self.solver = solver
        self.snapshots = None
        self.snapshot_times = None
        
    def run_simulation(self):
        """Run the complete simulation and store results."""
        print("Running finite difference simulation...")
        self.snapshots, self.snapshot_times = self.solver.solve(save_snapshots=True)
        print(f"Generated {len(self.snapshots)} snapshots")
        
    def create_static_plot(self, T_field, time_val, save_path=None, title_suffix=""):
        """Create a static temperature field plot."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create temperature contour plot
        contour = ax.contourf(self.solver.X, self.solver.Y, T_field, 
                             levels=50, cmap='turbo', extend='both')
        
        # Add colorbar
        cbar = plt.colorbar(contour, ax=ax, label='Temperature (°C)')
        
        # Add boundary rectangle
        x_rect = [x_min, x_max, x_max, x_min, x_min]
        y_rect = [y_min, y_min, y_max, y_max, y_min]
        ax.plot(x_rect, y_rect, color='black', linewidth=2)
        
        # Add heated boundary indicator
        ax.axvline(x=x_min, color='red', linewidth=4, alpha=0.7, label='Heated boundary')
        
        # Formatting
        ax.set_xlabel('x (m)', fontsize=12)
        ax.set_ylabel('y (m)', fontsize=12)
        ax.set_title(f'Temperature Distribution at t = {time_val:.2f} s{title_suffix}', fontsize=14)
        ax.set_aspect('equal')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            
    def create_animated_gif(self, filename='temperature_animation.gif', fps=10):
        """Create an animated GIF of the temperature evolution."""
        if self.snapshots is None:
            print("No simulation data available. Run simulation first.")
            return
            
        print("Creating temperature animation...")
        frames = []
        
        # Determine global temperature range for consistent colorbar
        T_min = np.min(self.snapshots)
        T_max = np.max(self.snapshots)
        
        for i, (T_field, time_val) in enumerate(zip(self.snapshots, self.snapshot_times)):
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Create temperature contour plot with fixed scale
            contour = ax.contourf(self.solver.X, self.solver.Y, T_field, 
                                 levels=50, cmap='turbo', vmin=T_min, vmax=T_max)
            
            # Add colorbar
            cbar = plt.colorbar(contour, ax=ax, label='Temperature (°C)')
            
            # Add boundary rectangle
            x_rect = [x_min, x_max, x_max, x_min, x_min]
            y_rect = [y_min, y_min, y_max, y_max, y_min]
            ax.plot(x_rect, y_rect, color='black', linewidth=2)
            
            # Add heated boundary indicator
            ax.axvline(x=x_min, color='red', linewidth=4, alpha=0.7)
            
            # Formatting
            ax.set_xlabel('x (m)', fontsize=12)
            ax.set_ylabel('y (m)', fontsize=12)
            ax.set_title(f'Temperature Distribution at t = {time_val:.2f} s', fontsize=14)
            ax.set_aspect('equal')
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            
            plt.tight_layout()
            
            # Convert to image array
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(image)
            plt.close()
            
            # Progress update
            if (i + 1) % 10 == 0:
                print(f"Generated frame {i + 1}/{len(self.snapshots)}")
        
        # Save as GIF
        imageio.mimsave(filename, frames, fps=fps)
        print(f"Animation saved to {filename}")
        
    def analyze_thermal_evolution(self, save_dir):
        """Analyze and plot thermal evolution over time."""
        if self.snapshots is None:
            print("No simulation data available. Run simulation first.")
            return
            
        print("Analyzing thermal evolution...")
        
        # Calculate statistics for each snapshot
        stats_over_time = {
            'time': self.snapshot_times,
            'max_temp': [],
            'min_temp': [],
            'avg_temp': [],
            'left_avg': [],
            'center_avg': [],
            'right_avg': []
        }
        
        for T_field in self.snapshots:
            stats = self.solver.analyze_temperature_distribution(T_field)
            stats_over_time['max_temp'].append(stats['max_temperature'])
            stats_over_time['min_temp'].append(stats['min_temperature'])
            stats_over_time['avg_temp'].append(stats['avg_temperature'])
            stats_over_time['left_avg'].append(stats['left_avg_temp'])
            stats_over_time['center_avg'].append(stats['center_avg_temp'])
            stats_over_time['right_avg'].append(stats['right_avg_temp'])
        
        # Create evolution plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Overall temperature evolution
        ax1.plot(stats_over_time['time'], stats_over_time['max_temp'], 
                label='Maximum', linewidth=2, color='red')
        ax1.plot(stats_over_time['time'], stats_over_time['avg_temp'], 
                label='Average', linewidth=2, color='blue')
        ax1.plot(stats_over_time['time'], stats_over_time['min_temp'], 
                label='Minimum', linewidth=2, color='green')
        
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Temperature (°C)')
        ax1.set_title('Overall Temperature Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Regional temperature evolution
        ax2.plot(stats_over_time['time'], stats_over_time['left_avg'], 
                label='Left region', linewidth=2, color='red')
        ax2.plot(stats_over_time['time'], stats_over_time['center_avg'], 
                label='Center region', linewidth=2, color='blue')
        ax2.plot(stats_over_time['time'], stats_over_time['right_avg'], 
                label='Right region', linewidth=2, color='green')
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Temperature (°C)')
        ax2.set_title('Regional Temperature Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'thermal_evolution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        return stats_over_time
        
    def create_summary_report(self, stats_over_time, save_dir):
        """Create a comprehensive summary report."""
        report_path = os.path.join(save_dir, 'finite_difference_analysis_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("FINITE DIFFERENCE HEAT DIFFUSION SIMULATION REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write("Configuration:\n")
            f.write(f"  Domain: {x_min} ≤ x ≤ {x_max} m, {y_min} ≤ y ≤ {y_max} m\n")
            f.write(f"  Grid resolution: {nx} × {ny} points\n")
            f.write(f"  Time steps: {nt} (dt = {dt:.6f} s)\n")
            f.write(f"  Thermal diffusivity (α): {alpha}\n")
            f.write(f"  Boundary temperature: {T_left}\n")
            f.write(f"  Initial temperature: {T_initial}\n\n")
            
            f.write("Numerical Parameters:\n")
            f.write(f"  Grid spacing: dx = {dx:.4f} m, dy = {dy:.4f} m\n")
            f.write(f"  Stability parameter: r_x = {r_x:.4f}, r_y = {r_y:.4f}\n")
            f.write(f"  Total r = {r_x + r_y:.4f} (stable if ≤ 0.5)\n\n")
            
            f.write("Final State Analysis:\n")
            f.write("-" * 30 + "\n")
            final_stats = {
                'max_temperature': stats_over_time['max_temp'][-1],
                'min_temperature': stats_over_time['min_temp'][-1],
                'avg_temperature': stats_over_time['avg_temp'][-1],
                'left_avg_temp': stats_over_time['left_avg'][-1],
                'center_avg_temp': stats_over_time['center_avg'][-1],
                'right_avg_temp': stats_over_time['right_avg'][-1]
            }
            
            for key, value in final_stats.items():
                f.write(f"  {key.replace('_', ' ').title()}: {value:.2f}°C\n")
            
            f.write(f"\nTemperature range: {final_stats['min_temperature']:.2f} - {final_stats['max_temperature']:.2f}°C\n")
        
        print(f"Summary report saved to {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Simulate 2D heat diffusion using finite differences')
    parser.add_argument('--output_dir', type=str, default='./results/finite_difference',
                      help='Directory to save simulation results')
    parser.add_argument('--create_gif', action='store_true', default=True,
                      help='Create animated GIF')
    parser.add_argument('--gif_fps', type=int, default=8,
                      help='Frames per second for GIF animation')
    parser.add_argument('--analyze', action='store_true', default=True,
                      help='Perform thermal analysis')
    
    args = parser.parse_args()
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    print("Initializing finite difference heat diffusion simulation...")
    print(f"Domain: {x_min} × {y_min} to {x_max} × {y_max} m")
    print(f"Grid: {nx} × {ny} points")
    print(f"Time steps: {nt}")
    
    # Initialize solver and simulator
    solver = HeatDiffusionSolver()
    simulator = HeatDiffusionSimulator(solver)
    
    # Run simulation
    simulator.run_simulation()
    
    # Create static plots at key time points
    print("\nGenerating static plots...")
    key_times = [0.1, 1.0, 5.0, 10.0, 25.0, 50.0]
    
    for target_time in key_times:
        # Find closest snapshot
        idx = np.argmin(np.abs(simulator.snapshot_times - target_time))
        actual_time = simulator.snapshot_times[idx]
        T_field = simulator.snapshots[idx]
        
        save_path = os.path.join(args.output_dir, f'temperature_t{target_time:.1f}.png')
        simulator.create_static_plot(T_field, actual_time, save_path)
    
    # Create animated GIF
    if args.create_gif:
        print("\nCreating animated visualization...")
        gif_path = os.path.join(args.output_dir, 'temperature_evolution.gif')
        simulator.create_animated_gif(gif_path, fps=args.gif_fps)
    
    # Perform analysis
    if args.analyze:
        print("\nPerforming thermal analysis...")
        stats_over_time = simulator.analyze_thermal_evolution(args.output_dir)
        simulator.create_summary_report(stats_over_time, args.output_dir)
    
    print(f"\nSimulation completed! Results saved to {args.output_dir}")
    print(f"Check the following files:")
    print(f"  - Static plots: temperature_t*.png")
    if args.create_gif:
        print(f"  - Animation: temperature_evolution.gif")
    if args.analyze:
        print(f"  - Analysis: thermal_evolution.png, temperature_profiles.png")
        print(f"  - Report: finite_difference_analysis_report.txt")

if __name__ == "__main__":
    main()
