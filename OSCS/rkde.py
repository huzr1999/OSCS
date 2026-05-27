import numpy as np

class RecursiveKDE:
    def __init__(self, grid_min, grid_max, num_grids, bandwidth, forgetting_factor, **kwargs):
        """
        Recursive Kernel Density Estimation (KDE) based on a fixed grid.
        
        Parameters:
        grid_min, grid_max: Grid range
        num_points: Number of grid points
        bandwidth: Kernel bandwidth
        forgetting_factor: Forgetting factor (0 < λ < 1)
        """

        self.grid_min = grid_min
        self.grid_max = grid_max

        self.grid = np.linspace(grid_min, grid_max, num_grids)
        self.bandwidth = bandwidth
        self.lambd = forgetting_factor
        
        # Initialize density estimate (uniform distribution)
        self.density_values = np.ones(num_grids) / (grid_max - grid_min)
        
        # Store history for visualization
        self.sample_history = []
        self.density_history = [self.density_values.copy()]
        self.update_count = 0
        
    def _gaussian_kernel(self, u, h):
        """Gaussian kernel function"""
        return np.exp(-0.5 * (u/h)**2) / (h * np.sqrt(2 * np.pi))
    
    def update(self, new_sample):
        """Update the density estimate using a new sample"""
        # Calculate the kernel function contribution of the new sample across all grid points
        kernel_contributions = self._gaussian_kernel(self.grid - new_sample, self.bandwidth)

        lambd = 1 - 1 / (self.update_count + 1) ** self.lambd  # Decay the forgetting factor over time

        
        # Recursive update: decay of old density + contribution of new sample
        self.density_values = (lambd * self.density_values + 
                             (1 - lambd) * kernel_contributions)
        
        # Normalize to ensure the integral is 1
        self._normalize_density()
        
        # Record history
        self.sample_history.append(new_sample)
        self.density_history.append(self.density_values.copy())
        self.update_count += 1
    
    def _normalize_density(self):
        """Normalize using numerical integration (trapezoidal rule)"""
        dx = self.grid[1] - self.grid[0]
        total_integral = np.trapezoid(self.density_values, self.grid)
        if total_integral > 0:
            self.density_values /= total_integral
    
    def evaluate(self, x_points=None):
        """Evaluate the density at given points, defaults to grid points"""
        if x_points is None:
            x_points = self.grid
        # Use linear interpolation to evaluate at arbitrary points
        return np.interp(x_points, self.grid, self.density_values)
    
    def get_current_density(self):
        """Return the current density estimate"""
        return self.grid, self.density_values
    
    def get_history(self):
        """Return historical data"""
        return self.sample_history, self.density_history

