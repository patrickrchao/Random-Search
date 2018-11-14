import OptimalInitialStepMagnitudes
import numpy as np

class optimization_step:
    optimal_step_dict = OptimalInitialStepMagnitudes.step_mag

    def __init__(self,step_params,optimizer_type):
       
        self.iteration = 1
        self.optimizer_type = optimizer_type
        self.step_function = step_params["STEP_FUNCTION"]

        function = step_params["FUNCTION"]

        if step_params["OPTIMAL"]:
            if optimizer_type == "ADAGRAD":
                self.init_step_magnitude = optimization_step.optimal_step_dict[optimizer_type][function]
                self.step_function = 'CONSTANT'
            else:
                self.init_step_magnitude = optimization_step.optimal_step_dict[optimizer_type][function][self.step_function]
        else:
            self.init_step_magnitude = step_params["INITIAL_STEP_MAGNITUDE"]

    def step_size(self):
        step_size_map = {
            "INV_SQ_ROOT":self.inv_sq_root_step(),
            "LOG": self.log_step(),
            "GEOMETRIC": self.geometric_step(),
            "CONSTANT": self.constant_step()
        }
        step = step_size_map[self.step_function]
        self.iteration += 1
        return step

    def inv_sq_root_step(self):
        return 1/np.sqrt(self.iteration+1)*self.init_step_magnitude

    def log_step(self):
        return 1/(np.log(self.iteration)+1)*self.init_step_magnitude

    def geometric_step(self):
        curr_step = self.init_step_magnitude
        amount_to_halve = 2
        counter = 0
        for i in range(self.iteration):
            counter +=1 
            if counter >= amount_to_halve:
                counter = 0
                curr_step /= 2
                amount_to_halve *= 2
        return curr_step

    def constant_step(self):
        return self.init_step_magnitude
