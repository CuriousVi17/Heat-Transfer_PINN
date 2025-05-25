import numpy as np
from fd_config import *

class HeatDiffusionSolver:
    """Finite difference solver for 2D heat diffusion equation."""
    
    def __init__(self):
        self.setup_grid()
        self.setup_initial_conditions()
        
    def setup_grid(self):
        """Initialize spatial and temporal grids."""
        self.x = np.linspace(x_min, x_max, nx)
        self.y = np.linspace(y_min, y_max, ny)
        self.t = np.linspace(t_min, t_max, nt)
        
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Temperature field (ny x nx)
        self.T = np.zeros((ny, nx))
        self.T_new = np.zeros_like(self.T)
        
    def setup_initial_conditions(self):
        """Set initial temperature distribution."""
        self.T.fill(T_initial)
        
    def apply_boundary_conditions(self):
        """Apply boundary conditions at each time step."""
        # Left boundary: constant temperature
        self.T[:, 0] = T_left
        
        # Right boundary: insulated (zero gradient)
        self.T[:, -1] = self.T[:, -2]
        
        # Top boundary: insulated (zero gradient)
        self.T[0, :] = self.T[1, :]
        
        # Bottom boundary: insulated (zero gradient)
        self.T[-1, :] = self.T[-2, :]
        
    def finite_difference_step(self):
        """Perform one time step using explicit finite difference scheme."""
        # Interior points using explicit finite difference
        for i in range(1, ny-1):
            for j in range(1, nx-1):
                # Second derivatives
                d2T_dx2 = (self.T[i, j+1] - 2*self.T[i, j] + self.T[i, j-1]) / dx**2
                d2T_dy2 = (self.T[i+1, j] - 2*self.T[i, j] + self.T[i-1, j]) / dy**2
                
                # Update temperature
                self.T_new[i, j] = self.T[i, j] + alpha * dt * (d2T_dx2 + d2T_dy2)
        
        # Copy new values
        self.T[1:-1, 1:-1] = self.T_new[1:-1, 1:-1]
        
        # Apply boundary conditions
        self.apply_boundary_conditions()
        
    def solve(self, save_snapshots=True):
        """Solve the heat diffusion equation over time."""
        if save_snapshots:
            # Store snapshots for animation
            snapshots = []
            snapshot_times = []
            
        print("Starting finite difference simulation...")
        print(f"Grid size: {nx} x {ny}")
        print(f"Time steps: {nt}")
        print(f"Total time: {t_max} s")
        
        for n in range(nt):
            # Perform one time step
            self.finite_difference_step()
            
            # Save snapshots for animation
            if save_snapshots and n % frame_interval == 0:
                snapshots.append(self.T.copy())
                snapshot_times.append(self.t[n])
                
            # Progress update
            if n % (nt // 10) == 0:
                progress = (n / nt) * 100
                print(f"Progress: {progress:.1f}% - Time: {self.t[n]:.2f} s")
        
        print("Simulation completed!")
        
        if save_snapshots:
            return np.array(snapshots), np.array(snapshot_times)
        else:
            return self.T
    
    def get_temperature_at_time(self, target_time):
        """Get temperature field at a specific time by running simulation."""
        current_time = 0.0
        time_step = 0
        
        # Reset to initial conditions
        self.setup_initial_conditions()
        
        while current_time < target_time and time_step < nt:
            self.finite_difference_step()
            current_time += dt
            time_step += 1
            
        return self.T.copy()
    
    def analyze_temperature_distribution(self, T_field=None):
        """Analyze thermal features of the temperature field."""
        if T_field is None:
            T_field = self.T
            
        stats = {
            'max_temperature': np.max(T_field),
            'min_temperature': np.min(T_field),
            'avg_temperature': np.mean(T_field),
        }
        
        # Analyze different regions
        # Left region (near heated boundary)
        left_region = T_field[:, :nx//5]
        stats['left_avg_temp'] = np.mean(left_region)
        
        # Center region
        center_start = nx//3
        center_end = 2*nx//3
        center_region = T_field[:, center_start:center_end]
        stats['center_avg_temp'] = np.mean(center_region)
        
        # Right region (far from heated boundary)
        right_region = T_field[:, 4*nx//5:]
        stats['right_avg_temp'] = np.mean(right_region)
        
        return stats
