from oracle import oracle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, ticker
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

from matplotlib.colors import LogNorm

def generate_plots(params,values,plot_type="Loss",optimizer_types=None):

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

    # If creating surface plot 
    if plot_type == "Surface" or plot_type == "Contour":
        function_oracle = oracle(oracle_params)
        normalize_variance = string_format(params["NORMALIZE_VARIANCE"])
        file_name += "{} ".format(normalize_variance)
        plot_title += "{} ".format(normalize_variance)
        assert (num_dimensions == 2 and values.shape[0] == 2), "Dimensions is not equal to 2, surface plot not available."

    if plot_type == "Surface":

        # Begin plotting
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        max_x = initialization_magnitude*1.1
        x = np.arange(-max_x,max_x,max_x/50)
        y = np.arange(-max_x,max_x,max_x/50)
        X, Y = np.meshgrid(x, y)

        function_oracle = oracle(oracle_params)

        evaluated_grid = np.array([function_oracle.query_function(np.array([[x],[y]])) for x,y in zip(np.ravel(X), np.ravel(Y))])

        Z = evaluated_grid.reshape(X.shape)
        evaluated_points = np.array([function_oracle.query_function(np.array([[x],[y]])) for x,y in values.T])
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False,alpha=0.4)
        lines = ax.scatter(values[0,:], values[1,:], evaluated_points,cmap='plasma',c=np.arange(0,values.shape[1]))


        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        plt.xlabel('X')
        plt.ylabel('Y')
        
    elif plot_type == "Contour":

        minimum = function_oracle.query_function(np.array([[0],[0]]))-1
        fig,ax = plt.subplots()
   
        max_x = initialization_magnitude*1.1
        x = np.linspace(-max_x-0.01,max_x+0.01,200)
        y = np.linspace(-max_x-0.01,max_x+0.01,200)
        X, Y = np.meshgrid(x, y)

        evaluated = np.array([function_oracle.query_function(np.array([[x],[y]])) for x,y in zip(np.ravel(X), np.ravel(Y))])-minimum
        Z = evaluated.reshape(X.shape)

        levels = np.array([function_oracle.query_function(np.array([[x],[0]])) for x in np.linspace(0.1,max_x*1.5,20)])-minimum
        ax.contourf(X, Y, Z, levels=levels,cmap="viridis",norm=LogNorm())
        ax.scatter(values[0,:],values[1,:],cmap='plasma', s=1.4,alpha=1,c=np.arange(0,values.shape[1]))

        for i in range(1, values.shape[1]):
            ax.annotate('', xy=(values[0,i], values[1,i]), xytext=(values[0,i-1], values[1,i-1]),
                           arrowprops={'arrowstyle': '-', 'color':'k',\
                                       'lw': 1,'alpha':0.5},
                           va='center', ha='center')

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.ylim(ymin=-max_x)
        plt.ylim(ymax=max_x)
        plt.xlim(xmin=-max_x)
        plt.xlim(xmax=max_x)

    else:

        average_values = np.mean(values,axis=0).squeeze()

        for i in range(average_values.shape[0]):
            plt.plot(average_values[i,:])

        plt.xlabel('Iterations')
        plt.ylabel(plot_type)

        if plot_type == "Distance from Origin":
            plt.ylim(ymin=0)
            plt.ylim(ymax=12)

        plt.legend(optimizer_types, loc='upper right', prop={'size': 6})

    # Add rest of file name
    file_name += "{} Step Sizes dims {} iters {} mag {} nu {} queries {} param {} cond num {}".\
    format(step_size,num_dimensions,iterations,initialization_magnitude,
            nu,queries,function_param,condition_num)
    
    # Update file path and plot title
    if step_params["OPTIMAL"]:
        plot_title += 'Convergence with {} Optimal Step Sizes Param {} '.\
            format(step_size,function_param)
        file_path += "Optimal/"
    else:
        file_name += " step mag {}".format(initial_step_magnitude)
        plot_title = 'Convergence with {} Step Sizes Param {} Init Step Mag {}'.\
            format(step_size,function_param,initial_step_magnitude)
    file_path += plot_type.replace(" ", "_")+"_Plots/"

    plt.title(plot_title)

    plt.savefig(file_path+file_name+".png", bbox_inches='tight',dpi=400)
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