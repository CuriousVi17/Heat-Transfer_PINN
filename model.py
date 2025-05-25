import tensorflow as tf
import numpy as np
from config import layers

class HeatPINN(tf.keras.Model):
    def __init__(self):
        super(HeatPINN, self).__init__()
        
        # Initialize the neural network layers
        self.model_layers = []
        for i in range(len(layers) - 2):
            self.model_layers.append(tf.keras.layers.Dense(
                layers[i+1], 
                activation=tf.keras.activations.tanh,
                kernel_initializer=tf.keras.initializers.GlorotNormal(),
                dtype=tf.float64
            ))
        
        # Output layer with no activation (linear) for temperature
        self.model_layers.append(tf.keras.layers.Dense(
            1,  # Single output: temperature T
            activation=None,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            dtype=tf.float64
        ))
        
    def call(self, X):
        """Forward pass through the network.
        X: input tensor of shape [batch_size, 3] representing (x, y, t)
        returns: output tensor of shape [batch_size, 1] representing T
        """
        H = X
        for layer in self.model_layers:
            H = layer(H)
        return H
    
    def get_temperature(self, x, y, t):
        """Get temperature field at given points.
        Returns: temperature T
        """
        X = tf.convert_to_tensor(np.stack([x, y, t], axis=1))
        T = self(X)
        return T
    
    def compute_pde_residuals(self, x, y, t, alpha):
        """Compute the PDE residuals for 2D heat equation.
        Heat equation: ∂T/∂t = α(∂²T/∂x² + ∂²T/∂y²)
        Returns: heat equation residual
        """
        x = tf.cast(x, tf.float64)
        y = tf.cast(y, tf.float64)
        t = tf.cast(t, tf.float64)
        alpha = tf.cast(alpha, tf.float64)
        
        points = tf.stack([x, y, t], axis=1)
        
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(points)
            
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch(points)
                T = self(points)
            
            # First-order derivatives
            T_x = tape1.gradient(T, points)[:, 0:1]
            T_y = tape1.gradient(T, points)[:, 1:2]
            T_t = tape1.gradient(T, points)[:, 2:3]
            
            # Second-order derivatives
            T_xx = tape2.gradient(T_x, points)[:, 0:1]
            T_yy = tape2.gradient(T_y, points)[:, 1:2]
        
        # Heat equation residual: ∂T/∂t - α(∂²T/∂x² + ∂²T/∂y²) = 0
        heat_residual = T_t - alpha * (T_xx + T_yy)
        
        return heat_residual