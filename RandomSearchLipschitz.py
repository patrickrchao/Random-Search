'''
 Patrick Chao 
 9/10/18 
 ML Research with Horia Mania

 Experimenting with changing lipschitz constant regimes
 '''
import argparse
import sys
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import LipschitzConstants
import random
import math

# CONSTANTS

iterations = LipschitzConstants.iterations
initialization_magnitude = LipschitzConstants.initialization_magnitude
nu = LipschitzConstants.nu
num_initializations = LipschitzConstants.num_initializations
num_gradient_calculations = LipschitzConstants.num_gradient_calculations

num_dimensions = 2
normalize_variance = False
step_size_type = 0

generate_surface_plots = False
generate_sigma_plots = True
normalize_variance_text = ""
noise_func_name = ""
verbose = False
show_plot = False
function = 0
param = 0
step_size_text = ""
function_text = ""
initial_step_magnitude = 0.2

quadratic_t = 5

# 1: Inverse Square Root, 2: log, 3: Geometric in step and iterations
def step_size(iteration):

    if step_size_type == 1:
        return 1/np.sqrt(iteration+1)*initial_step_magnitude
    elif step_size_type == 2:
        return np.log(iteration+1)*initial_step_magnitude
    elif step_size_type == 3:
        curr_step = initial_step_magnitude
        amount_to_halve = 2
        counter = 0
        for i in range(iteration):
            counter +=1 
            if counter >= amount_to_halve:
                counter = 0
                curr_step /= 2
                amount_to_halve *= 2
        return curr_step
        
    return 0


# Function Query 
# if function_num == 1, use a piecewise quadratic 
def query_function(function_num,x):
    norm = np.linalg.norm(x)
    # Quadratic
    if function_num  == 1:
        #Parameter should be on the order of 0.1 to 10
        if norm <= quadratic_t:
            return norm**2
        else:
            return param*(norm**2) +(-param+1)*(quadratic_t**2)
    elif function_num == 2:
        #Parameter should be on the order of 0.01 to 0.5

        if norm+param<0:
            return np.log((-1.0)*norm+param)
        elif (-1.0)*norm+param < 0:
            return np.log(norm+param)
        return max(np.log(norm+param),np.log((-1.0)*norm+param))
    return 0

# Randomly initializes the starting point of the optimization
# Takes the magnitude of the initialization as a parameter
# Optional argument for the quadrant the initialization should lie in for graphical purposes
def random_initialization(magnitude,quad=None):
    if not (quad==None):
        angle = np.random.rand()*(np.pi/2)+(quad-1)*(np.pi/2)
        x = np.cos(angle)
        y = np.sin(angle) 
        initialization = np.array([[x,y]]).flatten()*magnitude
        return initialization
    else:
        coords = np.array([random.normalvariate(0,1) for _ in range(num_dimensions)])
        r = np.linalg.norm(coords)
        return (np.array(coords)/r)*magnitude
   

# Function to minimize the quadratic form
# Takes in the function to query, random initialization, boolean for normalizing variance printing and plotting argument
def optimize(function,initialization=random_initialization(initialization_magnitude),normalize_variance = True,verbose = False,plot = False):
    # Initialize x values, representing current prediction for minima
    x_values = np.zeros((num_dimensions,iterations))
    x_values[:,0] = initialization

    curr_iter = 0
    if verbose:
        print_details(curr_iter,x_values)

    loss = np.zeros((iterations))
    sigma = np.zeros((iterations-1))
    loss[curr_iter] = query_function(function,x_values[:,curr_iter])

    # Iterate over number of iterations
    for curr_iter in range(1,iterations):
        # update represents the change from x_{j} to x_{j+1}
        update = np.zeros((1,num_dimensions))
        function_evals = np.zeros((num_gradient_calculations,2))
        for i in range(num_gradient_calculations):
            # delta is the random perturbation to x_{j} to calculate approximate gradient
            delta = np.random.normal(size=num_dimensions)
            # calculate the queried values x_{j} +/- nu*delta
            pos_delta_x = x_values[:,(curr_iter-1)]+nu*delta
            neg_delta_x = x_values[:,(curr_iter-1)]-nu*delta

            function_evals[i,0] = query_function(function,pos_delta_x)
            function_evals[i,1] = query_function(function,neg_delta_x)

            # accumulate the update and multiply by delta
            curr_update = (function_evals[i,0]-function_evals[i,1])*delta
            update = update + curr_update

        function_var = 1

        if normalize_variance:
            # Find variance of function evaluations
            #function_eval_mean = np.mean(function_evals)
            function_var = np.var(function_evals)

        curr_sigma = np.sqrt(function_var)
        sigma[curr_iter-1] = curr_sigma
        
        # update current 
        update = update.T.reshape(num_dimensions,1)
        alpha = step_size(curr_iter)
        x_values[:,curr_iter] = x_values[:,curr_iter-1]-update.T*alpha/(nu*curr_sigma*num_gradient_calculations)
        if verbose:
            if curr_iter % 25 == 0 :
                print_details(curr_iter,x_values)
        loss[curr_iter] = query_function(function,x_values[:,curr_iter])
    if verbose:
        print_details(function,curr_iter,x_values)
    if plot:
        plot_surface(function,x_values,loss,normalize_variance)
    return loss,sigma
    

# Plots the quadratic form and the calculated path
# Quadratic form is color coded using coolwarm mapping
# X values color coded using plasma, dark blue representing early and orange/yellow representing later values
def plot_surface(function,x_values,loss,normalize_variance):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
   
    max_x = initialization_magnitude*1.1 #np.abs(np.max(np.max(np.abs(x_values))))*1.2
    x = np.arange(-max_x,max_x,max_x/50)
    y = np.arange(-max_x,max_x,max_x/50)
    X, Y = np.meshgrid(x, y)
    evaluated = np.array([ query_function(function,np.array([[x],[y]])) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = evaluated.reshape(X.shape)
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False,alpha=0.4)
    lines = ax.scatter(x_values[0,:], x_values[1,:], loss,cmap='plasma',c=np.arange(0,x_values.shape[1]))
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    if normalize_variance:
        normalize_variance_text = "Normalized"
    else:
        normalize_variance_text = "Non-Normalized"
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('{} Convergence Path with {} Steps'.format(normalize_variance_text,step_size_text))
    plt.savefig("./Surface_Plots/Lipschitz/{} Function {} {} Step Sizes Dimensions {}  iters {} mag {} nu {}  calcs {}.png".
                    format(function_text,normalize_variance_text,step_size_text,num_dimensions,iterations,initialization_magnitude,nu,num_gradient_calculations) , bbox_inches='tight',dpi=400)
        
    if show_plot:
        plt.show()
        plt.close()

# Function for making the printing look nice
# If verbose argument in optimization is true, it will print this
def print_details(function,curr_iter,x_values):
    x = x_values[0,curr_iter]
    y = x_values[1,curr_iter]
    norm = np.linalg.norm(x_values[:,curr_iter])
    print("Iteration: {0:>5d}{4:3} (x,y): ({1:>6.2f}, {2:>6.2f}) {4:6} Norm: {3:6.2f}".format(curr_iter,x,y,norm,""))

def parse_args():
    global iterations,initialization_magnitude,nu,num_initializations,step_size_text
    global num_gradient_calculations,generate_surface_plots,verbose,show_plot
    global generate_sigma_plots,function, param, function_text, step_size_type

    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", default = LipschitzConstants.iterations,action="store",
                    help="number of iterations",dest="iterations",type=int)
    parser.add_argument("--init_mag", default = LipschitzConstants.initialization_magnitude,action="store",
                    help="initialization magnitude",dest="initialization_magnitude",type=int)
    parser.add_argument("--nu", default =  LipschitzConstants.nu,action="store",
                    help="standard deviation for query",dest="nu",type=float)
    parser.add_argument("--num_init", default = LipschitzConstants.num_initializations,action="store",
                    help="number initializations to process",dest="num_initializations",type=int)
    parser.add_argument("--num_samples", default = LipschitzConstants.num_gradient_calculations,action="store",
                    help="number of directions to sample to calculate approximate gradient",dest="num_gradient_calculations",type=int)
    parser.add_argument("--surface_plots", default = False, action="store_true",
                    help="boolean to generate surface plots rather than loss curves",dest="surface_plots")
    parser.add_argument("--sigma_plots", default = False, action="store_true",
                    help="boolean to generate sigma plots rather than loss curves",dest="sigma_plots")
    parser.add_argument("--verbose", default = False, action="store_true",
                    help="print out the (x,y) location and value",dest="verbose")
    parser.add_argument("--show_plot", default = False, action="store_true",
                    help="show the plot after generation",dest="show_plot")
    parser.add_argument("--function_type", default = 1, action="store",
                    help="1: Quadratic, 2: Log",dest="function_type",type=int)
    parser.add_argument("--step_function", default = 0, action="store",
                    help="1: Inverse Square Root, 2: log, 3: Geometric in step and iterations",dest="step_size_type",type=int)
    parser.add_argument("--num_dimensions", default = 2,action="store",
                    help="Number of dimensions",dest="num_dimensions",type=int)
    parser.add_argument("--param", default = 0,action="store",
                    help="Parameter for query function",dest="param",type=float)
    args = parser.parse_args()

    iterations = args.iterations
    initialization_magnitude = args.initialization_magnitude
    nu = args.nu
    num_initializations = args.num_initializations
    num_gradient_calculations = args.num_gradient_calculations
    generate_surface_plots = args.surface_plots
    generate_sigma_plots = args.sigma_plots
    verbose = args.verbose
    show_plot = args.show_plot
    function = args.function_type
    step_size_type = args.step_size_type
    num_dimensions = args.num_dimensions
    param = args.param
    if function == 1:
        function_text = "Quadratic"
    elif function == 2:
        function_text = "Log"

    if step_size_type == 1:
        step_size_text = "Inverse Square Root"
    elif step_size_type == 2:
        step_size_text = "Log"
    elif step_size_type == 3:
        step_size_text = "Geometric"

    if param == 0:
        if function == 1:
            param = 0.1
        else:
            param = 0.1

if __name__ == '__main__':
    parse_args()
    if generate_surface_plots:
        initialization = random_initialization(initialization_magnitude,quad = 2)
        optimize(function, initialization, normalize_variance=True,verbose=verbose,plot=True)
        optimize(function, initialization, normalize_variance=False,verbose=verbose,plot=True)

    else:
        
        loss = np.zeros((iterations,num_initializations,2))
        sigmas = np.zeros((iterations-1,num_initializations,2))
       # print_intialization()
        for trial in range(num_initializations):
            print("Initialization %d of %d"%(trial+1,num_initializations))
            
            initialization = random_initialization(initialization_magnitude)
            loss_norm,sigma_norm = optimize(function, initialization, normalize_variance=True,verbose=verbose)
            loss_non_norm,sigma_non_norm = optimize(function, initialization, normalize_variance=False,verbose=verbose)
            if generate_sigma_plots:
                sigmas[:,trial,0] = sigma_norm
                sigmas[:,trial,1] = sigma_non_norm
            
            loss[:,trial,0] = loss_norm
            loss[:,trial,1] = loss_non_norm



        if generate_sigma_plots:
            average_sigma = np.mean(sigmas,axis=1)
            plt.plot(average_sigma[:,0])
            plt.plot(average_sigma[:,1])
            plt.xlabel('Iterations')
            plt.ylabel('Sigma per Iteration')
            plt.ylim(ymin=0)
            plt.title('Sigma with {} Step Sizes'.format(step_size_text))
            plt.legend(['Normalized', 'Non-Normalized'], loc='upper right',prop={'size': 6})

            plt.savefig("./Sigma_Curves/Lipschitz/{} Function {} Step Sizes Dimensions {}  iters {} mag {} nu {} initializations {} calcs {}.png".
                    format(function_text,step_size_text,num_dimensions,iterations,initialization_magnitude,nu,num_initializations,num_gradient_calculations) , bbox_inches='tight',dpi=400)
            if show_plot:
                plt.show()


        plt.clf()
        average_loss = np.mean(loss,axis=1)
        plt.plot(average_loss[:,0])
        plt.plot(average_loss[:,1])
        
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.ylim(ymin=-0.1)
        #plt.ylim(ymax=5000)
        # if mat_size<=5:
        #     plt.ylim(ymax=1000)
        # elif mat_size<=10:
        #     plt.ylim(ymax=2500)
        plt.title('Convergence with {} Step Sizes'.format(step_size_text))
        plt.legend(['Normalized', 'Non-Normalized'], loc='upper right',prop={'size': 6})
        plt.savefig("./Loss_Curves/Lipschitz/{} Function {} Step Sizes Dimensions {}  iters {} mag {} nu {} initializations {} calcs {}.png".
                    format(function_text,step_size_text,num_dimensions,iterations,initialization_magnitude,nu,num_initializations,num_gradient_calculations) , bbox_inches='tight',dpi=400)
        if show_plot:
            plt.show()
    


    