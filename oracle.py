'''
 Patrick Chao
 11/8/18
 Optimization Research with Horia Mania

 Random Search Function Oracle

 '''

import numpy as np


class oracle:
    def __init__(self, params):
        self.function = params["FUNCTION"]
        self.condition_num = params["CONDITION_NUM"]
        self.function_param = params["FUNCTION_PARAM"]
        self.quadratic_threshold = 5
        self.params = params

    # Function Query
    # if function_num == 1, use quadratic
    # if function_num == 2, use log function
    def query_function(self, x):

        mat = np.diag((self.condition_num, 1))
        # Quadratic
        if self.function == "QUADRATIC":
            return oracle.norm(mat @ x, self.function_param, raise_to_power=False)

        elif self.function == "LOG":
            # Parameter should be on the order of 0.0001 to 0.1
            return np.log(np.linalg.norm(mat @ x) ** 2 + self.function_param)

        return 0

    # Function Derivative
    def derivative(self, x):

        mat = np.diag((self.condition_num, 1))

        if self.function == "QUADRATIC":
            # Parameter should be on the order of 0.1 to 10
            chain_rule = self.function_param * mat.T
            inner = (mat @ x) * np.power(np.abs(mat @ x), self.function_param - 2)
            return chain_rule @ inner
        elif self.function == "LOG":
            # Parameter should be on the order of 0.01 to 0.5
            return 2 * (mat.T @ mat @ x) / (np.linalg.norm(mat @ x) ** 2 + self.function_param)
        return 0

    # Empirical Function Derivative
    def empirical_derivative(self, x, num_queries, nu, output_function_evals=False):

        function_evals = np.zeros((num_queries, len(x)))
        start_angle = np.random.rand()*2*np.pi
        angles = np.linspace(start_angle, np.pi+start_angle, num_queries,endpoint=False)
        deltas = np.array([np.cos(angles), np.sin(angles)]).T
        pos_delta_x = x + nu * deltas
        neg_delta_x = x - nu * deltas
        function_evals[:, 0] = np.apply_along_axis(func1d=self.query_function, arr=pos_delta_x, axis=1)
        function_evals[:, 1] = np.apply_along_axis(func1d=self.query_function, arr=neg_delta_x, axis=1)
        function_at_point = self.query_function(x)
        function_diff = (function_evals[:, 0] - function_evals[:, 1]).reshape(-1, 1)
        # one half from sampling from both sides,
        # one half from expected value of each coordinate in delta
        grad = function_diff * deltas / nu
        total_grad = np.sum(grad, axis=0) / (num_queries)
        if output_function_evals:
            return total_grad, function_evals
        else:
            return total_grad

        # Empirically calculate lipschitz constant
        # Consider points of radius 1e-8 around the given point
        # Find the largest ratio of (f'(x)-f'(y))/(x-y)

    def lipschitz(self, x, gradient_type, num_samples=20, eps=1e-3, min_value=2):
        if gradient_type == "EMPIRICAL":
            nu = self.params["NU"]
            num_queries = self.params["QUERIES"]

        coords = np.random.randn(num_samples, len(x))
        r = np.linalg.norm(coords, axis=1).reshape(-1, 1)
        points_on_sphere = (coords / r) * eps
        shifted_points = points_on_sphere + x
        if gradient_type == "TRUE":
            derivatives = np.apply_along_axis(func1d=self.derivative, axis=1, arr=shifted_points)
            curr_derivative = self.derivative(x)
        else:
            true_derivatives = np.apply_along_axis(func1d=self.derivative, axis=1, arr=shifted_points)
            true_curr_derivative = self.derivative(x)
            true_derivative_diff = np.linalg.norm(true_curr_derivative - true_derivatives, axis=1)

            true_ratios = true_derivative_diff / eps
            true_lipschitz_constant = np.max(true_ratios)
            derivatives = np.apply_along_axis(func1d=self.empirical_derivative, axis=1, arr=shifted_points, nu=1e-3,
                                              num_queries=10)
            curr_derivative = self.empirical_derivative(x, nu=1e-3,
                                                        num_queries=100)

        derivative_diff = np.linalg.norm(curr_derivative - derivatives, axis=1)

        ratios = derivative_diff / eps
        lipschitz_constant = np.max(ratios)

        return max(lipschitz_constant, min_value)

    # Return the p-norm of a vector of an arbitrary p
    def norm(vec, p, raise_to_power=True):
        if np.all(vec == 0):
            return 0
        if raise_to_power:
            return (sum(np.abs(vec) ** p) ** (1. / p)).squeeze()
        else:
            return sum(np.abs(vec) ** p).squeeze()
