'''
 Patrick Chao
 11/8/18
 Optimization Research with Horia Mania

 Random Search Plotting

 '''

from oracle import oracle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, ticker
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter


# Values: the values to be plotted
# Plot type: Contour/Surface/Loss/Distance from Origin
# Optimizer Types: Adagrad, Normalized, Nonnormalized
def generate_plots(params, values, plot_type="Loss", optimizer_types=None):
    # Retrieve all params
    optimization_params = params['OPTIMIZATION']
    step_params = params['STEP']
    oracle_params = params['ORACLE']
    main_params = params['MAIN']

    function = string_format(oracle_params['FUNCTION'])
    step_size = string_format(step_params['STEP_FUNCTION'])
    num_dimensions = main_params['DIMENSIONS']
    iterations = optimization_params['ITERATIONS']
    initialization_magnitude = main_params['INITIALIZATION_MAGNITUDE']
    nu = optimization_params['NU']
    queries = optimization_params['QUERIES']
    function_param = oracle_params['FUNCTION_PARAM']
    condition_num = oracle_params['CONDITION_NUM']
    initial_step_magnitude = step_params['INITIAL_STEP_MAGNITUDE']


    # Begin file name
    file_name = "{} Function ".format(function)
    plot_title = "{} Function ".format(function)
    file_path = "./"
    max_x = 0
    # If creating surface plot or contour
    if plot_type == "Surface" or plot_type == "Contour":
        function_oracle = oracle(oracle_params)
        optimizer_type = string_format(params["OPTIMIZER_TYPE"])
        if step_size == "LIPSCHITZ" and optimizer_type != "NONNORMALIZE":
            step_size = "INV_SQ_ROOT"
        file_name += "{} ".format(optimizer_type)
        plot_title += "{} ".format(optimizer_type)
        assert (num_dimensions == 2 and values.shape[0] == 2), \
            "Dimensions is not equal to 2, surface plot not available."

    # Logic for creating surface plot
    if plot_type == "Surface":

        # Begin plotting
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        max_x = initialization_magnitude * 1.1
        plt.ylim(bottom=-max_x)
        plt.ylim(top=max_x)
        plt.xlim(left=-max_x)
        plt.xlim(right=max_x)

        # Calculate grid values
        x = np.linspace(-max_x - 0.01, max_x + 0.01, 200)
        y = np.linspace(-max_x - 0.01, max_x + 0.01, 200)
        X, Y = np.meshgrid(x, y)
        function_oracle = oracle(oracle_params)
        evaluated_grid = np.array(
            [function_oracle.query_function(np.array([[x], [y]])) for x, y in zip(np.ravel(X), np.ravel(Y))])
        Z = evaluated_grid.reshape(X.shape)

        # Evaluate function at given values
        evaluated_points = np.array([function_oracle.query_function(np.array([[x], [y]])) for x, y in values.T])
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False, alpha=0.4)
        lines = ax.scatter(values[0, :], values[1, :], evaluated_points, cmap='plasma', c=np.arange(0, values.shape[1]))

        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        plt.xlabel('X')
        plt.ylabel('Y')

    # Logic for creating contour plot
    elif plot_type == "Contour":

        minimum = function_oracle.query_function(np.array([[0], [0]])) - 1
        fig, ax = plt.subplots()
        max_x = initialization_magnitude * 1.1

        # Calculate grid values
        x = np.linspace(-max_x - 0.01, max_x + 0.01, 200)
        y = np.linspace(-max_x - 0.01, max_x + 0.01, 200)
        X, Y = np.meshgrid(x, y)
        evaluated = np.array(
            [function_oracle.query_function(np.array([[x], [y]])) for x, y in zip(np.ravel(X), np.ravel(Y))]) - minimum
        Z = evaluated.reshape(X.shape)

        levels = np.array(
            [function_oracle.query_function(np.array([[x], [x]])) for x in np.linspace(0.1, max_x, 20)]) - minimum

        ax.contourf(X, Y, Z, levels, cmap="viridis", norm=LogNorm(), extend='both')
        ax.scatter(values[0, :], values[1, :], cmap='plasma', s=1.4, alpha=1, c=np.arange(0, values.shape[1]))

        for i in range(1, values.shape[1]):
            ax.annotate('', xy=(values[0, i], values[1, i]), xytext=(values[0, i - 1], values[1, i - 1]),
                        arrowprops={'arrowstyle': '-', 'color': 'k', \
                                    'lw': 1, 'alpha': 0.5},
                        va='center', ha='center')

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.ylim(bottom=-max_x)
        plt.ylim(top=max_x)
        plt.xlim(left=-max_x)
        plt.xlim(right=max_x)

    else:

        average_values = np.median(values, axis=0).squeeze()
        for i in range(average_values.shape[0]):
            sd = np.sqrt(np.var(values[:, i, :], axis=0).squeeze() / values.shape[0])
            smooth_sd = gaussian_filter(sd, sigma=2)
            plt.plot(average_values[i, :])
            below = average_values[i, :] - smooth_sd * 1.96
            above = average_values[i, :] + smooth_sd * 1.96
            plt.fill_between(range(average_values.shape[1]), below, above, alpha=.2)

        plt.xlabel('Iterations')
        plt.ylabel(plot_type)

        if plot_type == "Distance from Origin":
            plt.ylim(bottom=0)
            plt.ylim(top=12)

        plt.legend(optimizer_types, loc='upper right', prop={'size': 6})

    # Add rest of file name
    file_name += "{} Step Sizes dims {} iters {} mag {} nu {} queries {} param {} cond num {}". \
        format(step_size, num_dimensions, iterations, initialization_magnitude,
               nu, queries, function_param, condition_num)
    # Update file path and plot title
    if step_params["OPTIMAL"]:
        plot_title += 'Convergence \n {} Optimal Step Sizes Param {}'. \
            format(step_size, function_param)
        file_path += "Optimal/"
    else:
        file_name += " step mag {}".format(initial_step_magnitude)
        plot_title += 'Convergence \n {} Step Sizes Param {} Init Step Mag {}'. \
            format(step_size, function_param, initial_step_magnitude)
    file_path += plot_type.replace(" ", "_") + "_Plots/"

    plt.title(plot_title)

    plt.savefig(file_path + file_name + ".png", dpi=400)
    plt.clf()


# Removes underscores and replaces them with spaces
# Properly capitalizes each word
def string_format(str):
    str = str.lower()
    if "-" in str:
        str = " ".join([s.capitalize() for s in str.split("-")])
    else:
        str = " ".join([s.capitalize() for s in str.split("_")])
    return str
