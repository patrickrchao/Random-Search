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

    # Function Derivative
    def derivative(self, x):
        # Quadratic
        mat = np.diag((self.condition_num, 1))

        if self.function == "QUADRATIC":
            # Parameter should be on the order of 0.1 to 10
            chain_rule = mat
            inner = (mat @ x) * np.power(np.abs(mat @ x), self.function_param - 2)
            return chain_rule @ inner
        elif self.function == "LOG":
            # Parameter should be on the order of 0.01 to 0.5
            return 2 * (mat.T @ mat @ x) / (np.linalg.norm(mat @ x)**2 + self.function_param)
        return 0

    # Function Derivative
    def lipschitz(self, x):
        # Quadratic
        return np.linalg.norm(self.derivative(x))

    # Function Query 
    # if function_num == 1, use quadratic
    # if function_num == 2, use log function
    def query_function(self, x):

        mat = np.diag((self.condition_num, 1))
        # Quadratic
        if self.function == "QUADRATIC":
            return oracle.norm(mat @ x, self.function_param) ** self.function_param

        elif self.function == "LOG":
            # Parameter should be on the order of 0.0001 to 0.1
            return np.log(np.linalg.norm(mat @ x)**2 + self.function_param)

        return 0

    # Return the p-norm of a vector of an arbitrary p
    def norm(vec, p):
        if np.all(vec==0):
            return 0
        return (sum(np.abs(vec) ** p) ** (1. / p)).squeeze()
