'''
 Patrick Chao 
 11/8/18 
 Optimization Research with Horia Mania

 Random Search Main File
 '''

import argparse
import numpy as np
import CONSTANTS
from optimization import optimizer
from oracle import oracle
from plotting import generate_plots
import random


# Initial logic for parsing arguments for optimization
def parse_args():
    parser = argparse.ArgumentParser()

    # MAIN PARAMETERS
    parser.add_argument("--num_init", default=CONSTANTS.num_initializations, action="store",
                        help="number initializations to process", dest="num_initializations", type=int)
    parser.add_argument("--num_dimensions", default=CONSTANTS.num_dimensions, action="store",
                        help="Number of dimensions", dest="num_dimensions", type=int)
    parser.add_argument("--init_mag", default=CONSTANTS.initialization_magnitude, action="store",
                        help="initialization magnitude", dest="initialization_magnitude", type=int)
    parser.add_argument("--surface_plots", default=False, action="store_true",
                        help="boolean to generate surface plots", dest="surface_plots")
    parser.add_argument("--contour_plots", default=False, action="store_true",
                        help="boolean to generate contour plots", dest="contour_plots")
    parser.add_argument("--search", default=False, action="store_true",
                        help="perform search for optimal initial step size", dest="search")

    # OPTIMIZATION PARAMETERS
    parser.add_argument("--iters", default=CONSTANTS.iterations, action="store",
                        help="number of iterations", dest="iterations", type=int)
    parser.add_argument("--nu", default=CONSTANTS.nu, action="store",
                        help="standard deviation for query", dest="nu", type=float)
    parser.add_argument("--queries", default=CONSTANTS.queries, action="store",
                        help="number of directions to sample to calculate approximate gradient (one in each direction)", \
                        dest="queries", type=int)

    # STEP FUNCTION PARAMETERS
    parser.add_argument("--step_function", default="INV_SQ_ROOT", action="store",
                        help="INV_SQ_ROOT: inverse square root, LOG: 1/log, GEOMETRIC: \
                    geometric in step and iterations, CONSTANT: constant over all iterations", dest="step_function")
    parser.add_argument("--init_step_mag", default=0.1, action="store",
                        help="The initial step size", dest="initial_step_mag", type=float)
    parser.add_argument("--optimal", default=False, action="store_true",
                        help="Plot with optimal initial step sizes", dest="optimal")

    # FUNCTION ORACLE PARAMETERS
    parser.add_argument("--function", default="LOG", action="store",
                        help="QUADRATIC: piecewise quadratic function of two quadratic forms, LOG: log(|x|+p)",
                        dest="function")
    parser.add_argument("--condition_num", default=CONSTANTS.condition_num, action="store",
                        help="Matrix Condition Num", dest="condition_num", type=float)
    parser.add_argument("--function_param", default=CONSTANTS.function_param, action="store",
                        help="Parameter for query function", dest="function_param", type=float)

    args = parser.parse_args()

    main_params = {
        'INITIALIZATIONS': args.num_initializations,
        'DIMENSIONS': args.num_dimensions,
        'INITIALIZATION_MAGNITUDE': args.initialization_magnitude,
        'SURFACE_PLOTS': args.surface_plots,
        'CONTOUR_PLOTS': args.contour_plots,
        'SEARCH': args.search
    }

    optimization_params = {
        'ITERATIONS': args.iterations,
        'NU': args.nu,
        'QUERIES': args.queries
    }

    step_params = {
        'STEP_FUNCTION': args.step_function,
        'INITIAL_STEP_MAGNITUDE': args.initial_step_mag,
        'OPTIMAL': args.optimal,
    }

    oracle_params = {
        'FUNCTION': args.function,
        'CONDITION_NUM': args.condition_num,
        'FUNCTION_PARAM': args.function_param
    }

    return main_params, optimization_params, step_params, oracle_params


# Randomly initializes the starting point of the optimization
# Takes the magnitude of the initialization as a parameter
# Optional argument for the quadrant the initialization should lie in for graphical purposes
def random_initialization(num_dimensions, magnitude, quadrant=None):
    if not (quadrant == None):
        angle = np.random.rand() * (np.pi / 2) + (quadrant - 1) * (np.pi / 2)
        x = np.cos(angle)
        y = np.sin(angle)
        initialization = np.array([[x, y]]).flatten() * magnitude
        return initialization
    else:
        # Random draws on a sphere
        coords = np.array([random.normalvariate(0, 1) for _ in range(num_dimensions)])
        r = np.linalg.norm(coords)
        return (np.array(coords) / r) * magnitude


# Computes loss and distance from origin for the optimizers
def optimize_trials(initializations, num_dimensions, initialization_magnitude, optimizers):
    loss = []
    dist_from_origin = []
    for trial in range(initializations):
        init = random_initialization(num_dimensions=num_dimensions, magnitude=initialization_magnitude)

        optimizer_losses = []
        optimizer_dists = []
        for curr_optimizer in optimizers:
            curr_loss, curr_dist, x_values = curr_optimizer.optimize(oracle, init)
            optimizer_losses.append(curr_loss)
            optimizer_dists.append(curr_dist)
            curr_optimizer.reset_optimization_step()
        loss.append(optimizer_losses)
        dist_from_origin.append(optimizer_dists)

    return np.array(loss), np.array(dist_from_origin)


if __name__ == '__main__':
    # Set optimizer types
    optimizer_types = ["NORMALIZE", "NONNORMALIZE", "ADAGRAD"]

    main_params, optimization_params, step_params, oracle_params = parse_args()

    params = {
        'MAIN': main_params,
        'OPTIMIZATION': optimization_params,
        'STEP': step_params,
        'ORACLE': oracle_params
    }

    num_dimensions = main_params["DIMENSIONS"]
    initializations = main_params["INITIALIZATIONS"]
    initialization_magnitude = main_params["INITIALIZATION_MAGNITUDE"]

    oracle = oracle(oracle_params)

    depth = CONSTANTS.depth
    sweep = CONSTANTS.sweep
    interval_shrinkage = CONSTANTS.interval_shrinkage

    # Step size parameter search
    if main_params["SEARCH"]:
        step_params["OPTIMAL"] = False
        optimal_step_sizes = []

        # Iterate over optimizer types
        for optimizer_type in optimizer_types:
            start_step = CONSTANTS.start_step
            end_step = CONSTANTS.end_step

            # Iterate over number of binary search bisections
            for curr_depth in range(depth):
                average_losses = []
                interval_length = end_step - start_step
                new_interval_length = interval_length * interval_shrinkage

                # Iterate over number of samples
                for curr_step_size in np.linspace(start_step, end_step, num=sweep):
                    step_params["INITIAL_STEP_MAGNITUDE"] = curr_step_size
                    optimizers = [optimizer(optimization_params, step_params,
                                            oracle_params, optimizer_type=optimizer_type)]
                    loss, _ = optimize_trials(initializations, num_dimensions, initialization_magnitude, optimizers)
                    loss = np.array(loss).squeeze(axis=1)
                    average_loss = np.median(loss[:, -5:])
                    average_losses.append(average_loss)

                curr_optimal_step = np.linspace(start_step, end_step, num=sweep)[np.argmin(average_losses)]
                # print(optimizer_type,curr_depth,curr_optimal_step)
                start_step = max(curr_optimal_step - new_interval_length / 2, 0.0001)
                end_step = curr_optimal_step + new_interval_length / 2

            optimal_step_sizes.append(curr_optimal_step)

        print("Function:", oracle_params["FUNCTION"], "Step Function:", step_params["STEP_FUNCTION"])
        print("Function Parameter:", oracle_params["FUNCTION_PARAM"], "Condition Number:",
              oracle_params["CONDITION_NUM"])
        print(optimal_step_sizes)

    else:

        optimizers = [optimizer(optimization_params, step_params, oracle_params, optimizer_type=optimizer_type)
                      for optimizer_type in optimizer_types]

        # Generate surface or contour plots if necessary
        if main_params["SURFACE_PLOTS"] or main_params["CONTOUR_PLOTS"]:

            init = random_initialization(num_dimensions=num_dimensions, magnitude=initialization_magnitude, quadrant=2)

            # Location for each iteration
            optimizer_x = []

            # Optimize
            for curr_optimizer in optimizers:
                _, _, x_values = curr_optimizer.optimize(oracle, init)
                optimizer_x.append(x_values)
            # Generate plots
            for curr_optimizer, x_values in zip(optimizers, optimizer_x):
                params["OPTIMIZER_TYPE"] = curr_optimizer.optimizer_type

                if main_params["SURFACE_PLOTS"]:
                    generate_plots(params, x_values, plot_type="Surface")
                if main_params["CONTOUR_PLOTS"]:
                    generate_plots(params, x_values, plot_type="Contour")

        # Calculate loss and distance from origin values
        loss, dist_from_origin = optimize_trials(initializations, num_dimensions, initialization_magnitude, optimizers)
        generate_plots(params, loss, "Loss", optimizer_types)
        generate_plots(params, dist_from_origin, "Distance from Origin", optimizer_types)
