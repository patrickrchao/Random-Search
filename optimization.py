'''
 Patrick Chao 
 11/8/18 
 Optimization Research with Horia Mania

 Random Search Optimization Calculations

 '''
from optimization_step import optimization_step
import numpy as np
import warnings


class optimizer:
    def __init__(self, optimization_params, step_params, oracle_params, optimizer_type):
        self.iterations = optimization_params["ITERATIONS"]
        self.nu = optimization_params["NU"]
        self.queries = optimization_params["QUERIES"]
        self.gradient_type = optimization_params["GRADIENT_TYPE"]
        self.optimizer_type = optimizer_type

        self.step_function = optimization_step(step_params, oracle_params, self.optimizer_type,self.gradient_type)

    def optimize(self, oracle, initialization):
        if self.optimizer_type == "ADAGRAD":
            return self.adagrad_optimize(oracle, initialization)
        else:
            return self.gradient_descent(oracle, initialization)

    # Takes in the function to query and initialization
    # Performs gradient descent with specified number of queries
    def gradient_descent(self, oracle, initialization):
        # Initialize x values, representing current prediction for minima
        num_dimensions = len(initialization)
        x_values = np.zeros((num_dimensions, self.iterations))
        x_values[:, 0] = initialization

        curr_iter = 0

        loss = np.zeros(self.iterations)
        dist_from_origin = np.zeros(self.iterations)

        loss[curr_iter] = oracle.query_function(initialization)
        dist_from_origin[curr_iter] = np.linalg.norm(initialization)

        # Iterate over number of iterations
        for curr_iter in range(1, self.iterations):
            curr_x = x_values[:, curr_iter - 1]
            alpha = self.step_function.step_size(curr_x)

            grad, sigma = self.get_derivative(curr_x, oracle)

            if self.optimizer_type == "NORMALIZE":
                if self.gradient_type == "TRUE":
                    grad = grad / np.linalg.norm(grad+1e-8)
                else:
                    grad = grad / sigma

            next_x = curr_x - alpha * grad

            # Add values
            x_values[:, curr_iter] = next_x

            loss[curr_iter] = oracle.query_function(next_x)
            dist_from_origin[curr_iter] = np.linalg.norm(next_x)

        return loss, dist_from_origin, x_values

    # Takes in the function to query and initialization
    # Performs adagrad with specified number of queries
    def adagrad_optimize(self, oracle, initialization):

        num_dimensions = len(initialization)
        # Initialize x values, representing current prediction for minima
        x_values = np.zeros((num_dimensions, self.iterations))
        x_values[:, 0] = initialization
        curr_iter = 0

        loss = np.zeros(self.iterations)
        dist_from_origin = np.zeros(self.iterations)
        loss[curr_iter] = oracle.query_function(x_values[:, curr_iter])
        dist_from_origin[curr_iter] = np.linalg.norm(x_values[:, curr_iter])

        cumulative_grad_magnitude = np.zeros(num_dimensions)

        # Iterate over number of iterations
        for curr_iter in range(1, self.iterations):
            curr_x = x_values[:, curr_iter - 1]
            alpha = self.step_function.step_size(curr_x)
            # Get the derivative
            curr_grad, sigma = self.get_derivative(curr_x.T, oracle)
            cumulative_grad_magnitude += np.square(curr_grad)

            # Update step with cumulative gradient magnitude
            next_x = curr_x - alpha * curr_grad / np.sqrt(cumulative_grad_magnitude+1e-8)

            # Add values
            x_values[:, curr_iter] = next_x
            loss[curr_iter] = oracle.query_function(next_x)
            dist_from_origin[curr_iter] = np.linalg.norm(next_x)

        return loss, dist_from_origin, x_values

    # Get the empirical or true derivative from the previous point
    # Uses the number of queries and nu as the sampling variance
    def get_derivative(self, prev_x, oracle):
        if self.gradient_type == "TRUE":
            grad = oracle.derivative(prev_x)
            return grad, None
        else:
            empirical_grad, function_evals = oracle.empirical_derivative(prev_x, self.queries, self.nu, output_function_evals=True)
            function_var = np.var(function_evals)
            sigma = np.sqrt(function_var)

            return empirical_grad, sigma

    def reset_optimization_step(self):
        self.step_function.reset_iteration()
