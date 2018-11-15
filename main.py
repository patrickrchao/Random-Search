'''
 Patrick Chao 
 11/8/18 
 ML Research with Horia Mania

 Random Search Main File

 '''
import argparse
import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
import CONSTANTS
from optimization import optimizer
from optimization_step import optimization_step
from oracle import oracle
from plotting import generate_plots
# import OptimalInitialStepMagnitudes
import random
# import math



def parse_args():
    parser = argparse.ArgumentParser()

    # MAIN PARAMETERS
    parser.add_argument("--num_init", default = CONSTANTS.num_initializations,action="store",
                    help="number initializations to process",dest="num_initializations",type=int)
    parser.add_argument("--num_dimensions", default = CONSTANTS.num_dimensions,action="store",
                    help="Number of dimensions",dest="num_dimensions",type=int)
    parser.add_argument("--init_mag", default = CONSTANTS.initialization_magnitude,action="store",
                    help="initialization magnitude",dest="initialization_magnitude",type=int)
    parser.add_argument("--surface_plots", default = False, action="store_true",
                    help="boolean to generate surface plots",dest="surface_plots")
    parser.add_argument("--contour_plots", default = False, action="store_true",
                    help="boolean to generate contour plots",dest="contour_plots")

    # OPTIMIZATION PARAMETERS
    parser.add_argument("--iters", default = CONSTANTS.iterations,action="store",
                    help="number of iterations",dest="iterations",type=int)
    parser.add_argument("--nu", default =  CONSTANTS.nu,action="store",
                    help="standard deviation for query",dest="nu",type=float)
    parser.add_argument("--queries", default = CONSTANTS.queries,action="store",
                    help="number of directions to sample to calculate approximate gradient (one in each direction)",\
                    dest="queries",type=int)
    #parser.add_argument("--adagrad", default = False, action="store_true",
    #                help="boolean to utilize adagrad",dest="adagrad")

    # STEP FUNCTION PARAMETERS
    parser.add_argument("--step_function", default = "INV_SQ_ROOT", action="store",
                    help="INV_SQ_ROOT: inverse square root, LOG: 1/log, GEOMETRIC: \
                    geometric in step and iterations, CONSTANT: constant over all iterations",dest="step_function")
    parser.add_argument("--init_step_mag", default = 0.1,action="store",
                    help="The initial step size",dest="initial_step_mag",type=float)
    parser.add_argument("--optimal", default = False,action="store_true",
                    help="Plot with optimal initial step sizes",dest="optimal")

    # FUNCTION ORACLE PARAMETERS
    parser.add_argument("--function", default = "LOG", action="store",
                    help="QUADRATIC: piecewise quadratic function of two quadratic forms, LOG: log(|x|+p)",dest="function")
    parser.add_argument("--condition_num", default = CONSTANTS.condition_num,action="store",
                    help="Matrix Condition Num",dest="condition_num",type=float)
    parser.add_argument("--function_param", default = CONSTANTS.function_param,action="store",
                    help="Parameter for query function",dest="function_param",type=float)
    
    
    args = parser.parse_args()

    main_params = {
    'INITIALIZATIONS' : args.num_initializations,
    'DIMENSIONS' : args.num_dimensions,
    'INITIALIZATION_MAGNITUDE' : args.initialization_magnitude,
    'SURFACE_PLOTS' : args.surface_plots,
    'CONTOUR_PLOTS' : args.contour_plots
    }

    optimization_params = {
    'ITERATIONS' : args.iterations,
    'NU' : args.nu,
    'QUERIES' : args.queries
    #'ADAGRAD' = args.adagrad
    }

    step_params = {
    'STEP_FUNCTION' : args.step_function,
    'INITIAL_STEP_MAGNITUDE' : args.initial_step_mag,
    'OPTIMAL' : args.optimal,
    #'ADAGRAD' : args.adagrad,
    'FUNCTION' : args.function
    }

    oracle_params = {
    'FUNCTION' : args.function,
    'CONDITION_NUM' : args.condition_num,
    'FUNCTION_PARAM' : args.function_param
    }


    return main_params,optimization_params,step_params,oracle_params
    

    # Randomly initializes the starting point of the optimization
    # Takes the magnitude of the initialization as a parameter
    # Optional argument for the quadrant the initialization should lie in for graphical purposes
def random_initialization(num_dimensions,magnitude,quadrant=None):
    if not (quadrant == None):
        angle = np.random.rand()*(np.pi/2)+(quadrant-1)*(np.pi/2)
        x = np.cos(angle)
        y = np.sin(angle) 
        initialization = np.array([[x,y]]).flatten()*magnitude
        return initialization
    else:
        coords = np.array([random.normalvariate(0,1) for _ in range(num_dimensions)])
        r = np.linalg.norm(coords)
        return (np.array(coords)/r)*magnitude


if __name__ == '__main__':
    main_params,optimization_params,step_params,oracle_params = parse_args()

    params = {
    'MAIN':main_params,
    'OPTIMIZATION':optimization_params,
    'STEP': step_params,
    'ORACLE':oracle_params
    }

    num_dimensions = main_params["DIMENSIONS"]
    initializations = main_params["INITIALIZATIONS"]
    initialization_magnitude = main_params["INITIALIZATION_MAGNITUDE"]

    oracle = oracle(oracle_params)
    optimizer_types = ["NORMALIZE", "NONNORMALIZE", "ADAGRAD"]
    optimizers = [optimizer(optimization_params, step_params, optimizer_type=optimizer_type) for optimizer_type in
                  optimizer_types]
    if main_params["SURFACE_PLOTS"] or main_params["CONTOUR_PLOTS"]:

        init = random_initialization(num_dimensions=num_dimensions,magnitude=initialization_magnitude,quadrant = 2)

        optimizer_x = []

        for curr_optimizer in optimizers:
            _, _, x_values = curr_optimizer.optimize(oracle, init)
            optimizer_x.append(x_values)


        for curr_optimizer, x_values in zip(optimizers,optimizer_x):
            params["NORMALIZE_VARIANCE"] = curr_optimizer.optimizer_type
            #if main_params["SURFACE_PLOTS"]:
            generate_plots(params, x_values, plot_type="Surface")
            if main_params["CONTOUR_PLOTS"]:
                generate_plots(params, x_values, plot_type="Contour")

    iterations = optimization_params["ITERATIONS"]

    outputs = []
    loss = []
    dist_from_origin = []
    for trial in range(initializations):

        init = random_initialization(num_dimensions=num_dimensions,magnitude=initialization_magnitude)

        optimizer_losses = []
        optimizer_dists = []
        for curr_optimizer in optimizers:
            curr_loss,curr_dist, x_values = curr_optimizer.optimize(oracle, init)
            optimizer_losses.append(curr_loss)
            optimizer_dists.append(curr_dist)

        loss.append(optimizer_losses)
        dist_from_origin.append(optimizer_dists)


    loss = np.array(loss)
    dist_from_origin = np.array(dist_from_origin)
    generate_plots(params,loss,"Loss",optimizer_types)
    generate_plots(params,dist_from_origin,"Distance from Origin",optimizer_types)

    