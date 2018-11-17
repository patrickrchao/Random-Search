'''
 Patrick Chao 
 11/8/18 
 ML Research with Horia Mania

 Random Search Optimize

 '''
from optimization_step import optimization_step
import numpy as np


class optimizer:
    def __init__(self, optimization_params, step_params, oracle_params, optimizer_type):
        self.iterations = optimization_params["ITERATIONS"]
        self.nu = optimization_params["NU"]
        self.queries = optimization_params["QUERIES"]
        self.optimizer_type = optimizer_type

        self.step_function = optimization_step(step_params,oracle_params, self.optimizer_type)

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

        loss[curr_iter] = oracle.query_function(x_values[:, curr_iter])
        dist_from_origin[curr_iter] = np.linalg.norm(x_values[:, curr_iter])

        # Iterate over number of iterations
        for curr_iter in range(1, self.iterations):
            # update represents the change from x_{j} to x_{j+1}
            update = np.zeros((1, num_dimensions))
            function_evals = np.zeros((self.queries, num_dimensions))
            for i in range(self.queries):
                # delta is the random perturbation to x_{j} to calculate approximate gradient
                delta = np.random.normal(size=num_dimensions)
                # calculate the queried values x_{j} +/- nu*delta
                pos_delta_x = x_values[:, (curr_iter - 1)] + self.nu * delta
                neg_delta_x = x_values[:, (curr_iter - 1)] - self.nu * delta

                function_evals[i, 0] = oracle.query_function(pos_delta_x)
                function_evals[i, 1] = oracle.query_function(neg_delta_x)

                # accumulate the update and multiply by delta
                curr_update = (function_evals[i, 0] - function_evals[i, 1]) * delta
                update = update + curr_update

            function_var = np.var(function_evals)
            curr_sigma = np.sqrt(function_var)
            if not (self.optimizer_type == "NORMALIZE"):
                curr_sigma = 1

            # update current 
            update = update.T.reshape(num_dimensions, 1)
            alpha = self.step_function.step_size()
            x_values[:, curr_iter] = x_values[:, curr_iter - 1] - update.T * alpha / (
                    self.nu * curr_sigma * self.queries)

            loss[curr_iter] = oracle.query_function(x_values[:, curr_iter])
            dist_from_origin[curr_iter] = np.linalg.norm(x_values[:, curr_iter])

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
            function_evals = np.zeros((self.queries, 2))
            for i in range(self.queries):
                curr_grad = oracle.derivative(x_values[:, curr_iter - 1])
                cumulative_grad_magnitude += np.square(curr_grad)

            alpha = self.step_function.step_size()
            x_values[:, curr_iter] = x_values[:, curr_iter - 1] - alpha / np.sqrt(cumulative_grad_magnitude) * curr_grad

            loss[curr_iter] = oracle.query_function(x_values[:, curr_iter])
            dist_from_origin[curr_iter] = np.linalg.norm(x_values[:, curr_iter])

        # if plot:
        #     plot_surface(function,x_values,loss,normalize_variance)

        return loss, dist_from_origin, x_values

    def reset_optimization_step(self):
        self.step_function.reset_iteration()
