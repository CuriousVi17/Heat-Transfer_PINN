import tensorflow as tf
import numpy as np
import os
import time
from model import HeatPINN
from utils import (
    generate_domain_points, generate_boundary_points, 
    generate_initial_points, plot_temperature_field, create_directories
)
from config import (
    alpha, T_left, T_initial, t_min, t_max,
    n_domain, n_boundary, n_initial,
    epochs, batch_size, learning_rate,
    sample_every, plot_every, model_save_path, results_path
)

class HeatSolver:
    def __init__(self):
        self.model = HeatPINN()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Create directories for saving results
        create_directories()
        
        # Initialize loss tracker
        self.loss_history = {
            'total_loss': [],
            'pde_loss': [],
            'boundary_loss': [],
            'initial_loss': [],
        }
        
        # For sampling during training
        self.current_epoch = 0
    
    def compute_boundary_loss(self, boundary_points, boundary_types):
        """Compute loss for boundary conditions."""
        x, y, t = boundary_points[:, 0], boundary_points[:, 1], boundary_points[:, 2]
        
        # Get predicted temperatures
        points_tf = tf.convert_to_tensor(boundary_points, dtype=tf.float64)
        T_pred = self.model(points_tf)[:, 0]
        
        # Initialize loss
        loss = 0.0
        
        # Left boundary (heated): T = T_left = 100°C
        mask_left = boundary_types == 0
        T_left_loss = tf.reduce_mean(tf.square(T_pred[mask_left] - T_left))
        loss += T_left_loss
        
        # Right, top, and bottom boundaries (insulated): ∂T/∂n = 0
        # For insulated boundaries, we compute the normal derivative and set it to zero
        for boundary_id in [1, 2, 3]:  # right, bottom, top
            mask_boundary = boundary_types == boundary_id
            if tf.reduce_sum(tf.cast(mask_boundary, tf.float64)) > 0:
                boundary_points_masked = tf.boolean_mask(points_tf, mask_boundary)
                
                with tf.GradientTape() as tape:
                    tape.watch(boundary_points_masked)
                    T_boundary = self.model(boundary_points_masked)
                
                # Compute temperature gradients
                T_grad = tape.gradient(T_boundary, boundary_points_masked)
                
                if boundary_id == 1:  # Right boundary: ∂T/∂x = 0
                    normal_grad = T_grad[:, 0:1]
                elif boundary_id == 2:  # Bottom boundary: ∂T/∂y = 0
                    normal_grad = T_grad[:, 1:2]
                else:  # Top boundary: ∂T/∂y = 0
                    normal_grad = -T_grad[:, 1:2]  # Negative because normal points outward
                
                insulated_loss = tf.reduce_mean(tf.square(normal_grad))
                loss += 0.1 * insulated_loss  # Weight for insulated boundary condition
        
        return loss
    
    def compute_initial_loss(self, initial_points):
        """Compute loss for initial conditions."""
        # Get predicted temperatures at t=0
        points_tf = tf.convert_to_tensor(initial_points, dtype=tf.float64)
        T_pred = self.model(points_tf)[:, 0]
        
        # Initial condition: T = T_initial = 10°C everywhere at t=0
        T_loss = tf.reduce_mean(tf.square(T_pred - T_initial))
        
        return T_loss
    
    def compute_pde_loss(self, domain_points):
        """Compute loss for PDE residuals."""
        domain_points = tf.convert_to_tensor(domain_points, dtype=tf.float64)
        x, y, t = domain_points[:, 0], domain_points[:, 1], domain_points[:, 2]
        
        # Compute PDE residuals using the heat equation
        heat_residual = self.model.compute_pde_residuals(x, y, t, alpha)
        
        # Compute mean squared error of residuals
        pde_loss = tf.reduce_mean(tf.square(heat_residual))
        
        return pde_loss
    
    @tf.function
    def train_step(self, domain_points, boundary_points, boundary_types, initial_points):
        """Perform one training step."""
        with tf.GradientTape() as tape:
            # Compute losses
            pde_loss = self.compute_pde_loss(domain_points)
            boundary_loss = self.compute_boundary_loss(boundary_points, boundary_types)
            initial_loss = self.compute_initial_loss(initial_points)
            
            # Define loss weights
            w_pde = 1.0
            w_boundary = 100.0
            w_initial = 100.0
            
            # Total loss
            total_loss = (w_pde * pde_loss + 
                         w_boundary * boundary_loss + 
                         w_initial * initial_loss)
        
        # Compute gradients and update parameters
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return total_loss, pde_loss, boundary_loss, initial_loss
    
    def train(self):
        """Train the PINN model."""
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            
            # Generate training points
            domain_points = generate_domain_points(n_domain, t_min, t_max)
            boundary_points, boundary_types = generate_boundary_points(n_boundary, t_min, t_max)
            initial_points = generate_initial_points(n_initial)
            
            # Train step
            total_loss, pde_loss, boundary_loss, initial_loss = self.train_step(
                domain_points, boundary_points, boundary_types, initial_points
            )
            
            # Record losses
            self.loss_history['total_loss'].append(total_loss.numpy())
            self.loss_history['pde_loss'].append(pde_loss.numpy())
            self.loss_history['boundary_loss'].append(boundary_loss.numpy())
            self.loss_history['initial_loss'].append(initial_loss.numpy())
            
            # Print progress
            if epoch % 200 == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch}/{epochs}, Time: {elapsed:.2f}s")
                print(f"  Total Loss: {total_loss:.6f}")
                print(f"  PDE Loss: {pde_loss:.6f}, Boundary Loss: {boundary_loss:.6f}")
                print(f"  Initial Loss: {initial_loss:.6f}")
                
            # Save model checkpoint
            if epoch % sample_every == 0:
                save_path = os.path.join(model_save_path, f"model_epoch_{epoch}")
                self.model.save_weights(save_path)
                
            # Generate plots for visualization
            if epoch % plot_every == 0:
                # Save temperature field at different time steps
                for t in [0.1, 1.0, 5.0, 10.0, 25.0]:
                    plot_temperature_field(
                        self.model, t,
                        save_path=os.path.join(results_path, 'temperature', f"temp_t{t}_epoch{epoch}.png")
                    )
        
        # Save final model
        self.model.save_weights(os.path.join(model_save_path, "model_final"))
        print(f"Training completed in {time.time() - start_time:.2f} seconds")
        
        # Plot final loss history
        self.plot_loss_history()
    
    def plot_loss_history(self):
        """Plot the training loss history."""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(15, 10))
        
        # Plot total loss
        plt.subplot(2, 2, 1)
        plt.semilogy(self.loss_history['total_loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.title('Total Loss')
        plt.grid(True)
        
        # Plot PDE loss
        plt.subplot(2, 2, 2)
        plt.semilogy(self.loss_history['pde_loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.title('PDE Loss')
        plt.grid(True)
        
        # Plot boundary loss
        plt.subplot(2, 2, 3)
        plt.semilogy(self.loss_history['boundary_loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.title('Boundary Loss')
        plt.grid(True)
        
        # Plot initial loss
        plt.subplot(2, 2, 4)
        plt.semilogy(self.loss_history['initial_loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.title('Initial Loss')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_path, 'loss_history.png'), dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    solver = HeatSolver()
    solver.train()
