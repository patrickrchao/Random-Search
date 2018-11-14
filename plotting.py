from oracle import oracle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def generate_plots(params,values,plot_type="Loss",optimizer_types=None):

    # Retrieve all params
    optimization_params = params['OPTIMIZATION']
    step_params = params['STEP']
    oracle_params = params['ORACLE']
    main_params = params['MAIN']

    function = oracle_params['FUNCTION']
    step_size = step_params['STEP_FUNCTION']
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
    file_path = "./"

    # If creating surface plot 
    if plot_type == "Surface" or plot_type == "Contour":
        function_oracle = oracle(oracle_params)
        normalize_variance = params["NORMALIZE_VARIANCE"]
        file_name += "{} ".format(normalize_variance)

    if plot_type == "Surface":
        
        if num_dimensions != 3:
            print("Dimensions is not equal to 2, surface plot not available.")
            return

        # Update file path and file name    
        file_path += "Surface_Plots/"
        

        # Begin plotting
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        max_x = max(initialization_magnitude*1.1,np.abs(np.max(np.max(np.abs(values))))*1.2)
        x = np.arange(-max_x,max_x,max_x/50)
        y = np.arange(-max_x,max_x,max_x/50)
        X, Y = np.meshgrid(x, y)

        function_oracle = oracle(step_params)

        evaluated = np.array([function_oracle.query_function(np.array([[x],[y]])) for x,y in zip(np.ravel(X), np.ravel(Y))])
        Z = evaluated.reshape(X.shape)
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False,alpha=0.4)
        lines = ax.scatter(values[0,:], values[1,:], loss,cmap='plasma',c=np.arange(0,values.shape[1]))
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        plt.xlabel('X')
        plt.ylabel('Y')
        
    if plot_type == "Contour":

        file_path += "Contour_Plots/"

        fig,ax = plt.subplots()
   
        max_x = max(initialization_magnitude*1.1,np.abs(np.max(np.max(np.abs(values))))*1.2)
        x = np.arange(-max_x,max_x,max_x/50)
        y = np.arange(-max_x,max_x,max_x/50)
        X, Y = np.meshgrid(x, y)

        evaluated = np.array([function_oracle.query_function(np.array([[x],[y]])) for x,y in zip(np.ravel(X), np.ravel(Y))])
        Z = evaluated.reshape(X.shape)
        ax.contour(X, Y, Z, levels=np.logspace(-3, 3, 25), cmap='jet')
        #surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
    #                           linewidth=0, antialiased=False,alpha=0.4)
        for i in range(1, values.shape[1]):
            ax.annotate('', xy=(values[0,i], values[1,i]), xytext=(values[0,i-1], values[1,i-1]),
                           arrowprops={'arrowstyle': '->', 'color': 'b', 'lw': 1},
                           va='center', ha='center')

        plt.xlabel('X')
        plt.ylabel('Y')

    else:

        average_values = np.mean(values,axis=0).squeeze()

        for i in range(average_values.shape[0]):
            plt.plot(average_values[i,:])

        plt.xlabel('Iterations')
        plt.ylabel(plot_type)

        if plot_type == "Distance from Origin":
            plt.ylim(ymin=0)
            plt.ylim(ymax=12)


    # Add rest of file name
    file_name = "{} Step Sizes dims {} iters {} mag {} nu {} queries {} param {} cond num {}".\
    format(step_size,num_dimensions,iterations,initialization_magnitude,
            nu,queries,function_param,condition_num)
    
    # Update file path and plot title
    if step_params["OPTIMAL"]:
        plot_title = '{} Function Convergence with {} Optimal Step Sizes Param {} '.\
            format(function,step_size,function_param)
        file_path += "Optimal/"
    else:
        file_name += " step mag {}".format(initial_step_magnitude)
        plot_title = '{} Function Convergence with {} Step Sizes Param {} Init Step Mag {}'.\
            format(function,step_size,function_param,initial_step_magnitude)
    file_path += plot_type.replace(" ", "_")+"_Curves/"

    plt.title(plot_title)
    plt.legend(optimizer_types,loc='upper right',prop={'size':6})
    plt.savefig(file_path+file_name+".png", bbox_inches='tight',dpi=400)

