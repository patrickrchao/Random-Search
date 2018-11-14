import numpy as np

class oracle:
    def __init__(self,params):
        self.function = params["FUNCTION"]
        self.condition_num = params["CONDITION_NUM"]
        self.function_param = params["FUNCTION_PARAM"]
        
        self.quadratic_threshold = 5
        
    # Function Derivative
    def derivative(self,x):
        # Quadratic
        if self.function == "QUADRATIC":
            #Parameter should be on the order of 0.1 to 10
            if x <= self.quadratic_threshold:
                return 2*x
            else:
                return 2*self.function_param*x
        elif self.function == "LOG":
            #Parameter should be on the order of 0.01 to 0.5

            if x < 0:
                return 1/(self.function_param + x)
            elif x >= 0:
                return -1/(self.function_param - x)
        return 0

    # Function Query 
    # if function_num == 1, use a piecewise quadratic 
    # if function_num == 2, use log function
    def query_function(self,x):

        norm = np.linalg.norm(x)
        # Quadratic
        if self.function == "QUADRATIC":
            #Parameter should be on the order of 0.1 to 10
            if norm <= self.quadratic_threshold:
                return norm**2
            else:
                return self.function_param*(norm**2) +(-self.function_param+1)*(self.quadratic_threshold**2)

        elif self.function == "LOG":

            #Parameter should be on the order of 0.01 to 0.5
            mat = np.diag((self.condition_num,1))
            return np.log(np.linalg.norm(mat@x)+self.function_param)
            
        return 0