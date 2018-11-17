import numpy as np
import pandas as pd
import warnings


class optimization_step:
    optimal_step_dict = pd.read_csv("OptimalInitialStepSizes.csv")
    optimal_step_dict = optimal_step_dict.set_index(['FUNCTION', 'PARAM', 'CONDITION_NUM',
                                                     'OPTIMIZER_TYPE', 'STEP_FUNCTION_TYPE'])

    def __init__(self, step_params, oracle_params, optimizer_type):

        self.iteration = 1
        self.optimizer_type = optimizer_type
        self.step_function = step_params["STEP_FUNCTION"]

        function_type = oracle_params["FUNCTION"]
        condition_num = oracle_params["CONDITION_NUM"]
        function_param = oracle_params["FUNCTION_PARAM"]
        if step_params["OPTIMAL"]:
            index = function_type, function_param, condition_num, optimizer_type, self.step_function
            try:
                self.init_step_magnitude = optimization_step.optimal_step_dict.loc[index].values[0]
            except:
                warnings.warn("Optimal step size not found for \n"+str(index))
                self.init_step_magnitude = step_params["INITIAL_STEP_MAGNITUDE"]

        else:
            self.init_step_magnitude = step_params["INITIAL_STEP_MAGNITUDE"]

    def reset_iteration(self):
        self.iteration = 1

    def step_size(self):
        step_size_map = {
            "INV_SQ_ROOT": self.inv_sq_root_step(),
            "LOG": self.log_step(),
            "GEOMETRIC": self.geometric_step(),
            "CONSTANT": self.constant_step()
        }
        step = step_size_map[self.step_function]
        self.iteration += 1
        return step

    def inv_sq_root_step(self):
        return 1 / np.sqrt(self.iteration + 1) * self.init_step_magnitude

    def log_step(self):
        return 1 / (np.log(self.iteration) + 1) * self.init_step_magnitude

    def geometric_step(self):
        curr_step = self.init_step_magnitude
        amount_to_halve = 2
        counter = 0
        for i in range(self.iteration):
            counter += 1
            if counter >= amount_to_halve:
                counter = 0
                curr_step /= 2
                amount_to_halve *= 2
        return curr_step

    def constant_step(self):
        return self.init_step_magnitude
