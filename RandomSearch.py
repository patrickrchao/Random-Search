'''
 Patrick Chao 
 5/12/18 
 ML Research with Horia Mania
 '''
import argparse
import sys
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import Constants
import random
import math

# CONSTANTS
mat_size = Constants.mat_size 
alpha = Constants.alpha 
iterations = Constants.iterations
initialization_magnitude = Constants.initialization_magnitude
nu = Constants.nu
max_noise = Constants.max_noise
num_initializations = Constants.num_initializations
num_gradient_calculations = Constants.num_gradient_calculations
normalize_variance = False

generate_surface_plots = False
normalize_variance_text = ""
noise_func_name = ""
verbose = False
show_plot = False

no_cov = lambda x: 0
inc_quad_cov = lambda x: x**2*max_noise/(iterations**2)
dec_quad_cov = lambda x: (iterations-x)**2*max_noise/(iterations**2)
constant_cov = lambda x: max_noise/3
# Takes in dimension of matrix as input
# Generates a random positive definite matrix with eigenvalues at least 0.1
def generate_pd(size):
    rand_mat = (np.random.rand(size,size)-0.5)*2#; % generate a random n x n matrix
    pd_mat = 0.5*(rand_mat@rand_mat.T)
    eigenvalues = np.linalg.eigvals(pd_mat)
    if size ==  2:
        if not np.all(eigenvalues > 0.5):
            return generate_pd(size)

    else:
        if not np.all(eigenvalues > 0.01):
            return generate_pd(size)
    return pd_mat

# Given a matrix, and input vector, returns x.T A x 
# Optional argument for adding noise, takes in the variance of noise 1d array
# Default is no noise
def query_function(mat,x,noise_variance=np.zeros((1,1))):
    noise = np.random.multivariate_normal(np.array([0]), noise_variance, 1)
    function_eval =  x.T@mat@x
    output = function_eval + noise.T
    return output

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
        coords = np.array([random.normalvariate(0,1) for _ in range(mat_size)])
        r = np.linalg.norm(coords)
        return np.array(coords)/r*magnitude

        # initialization = np.random.rand(mat_size,)
        # scaled_initialization = initialization/np.linalg.norm(initialization)*magnitude
        # return scaled_initialization
   

# Function to minimize the quadratic form
# Takes in the quadratic form matrix, random initialization, covariance function per iteration, printing and plotting argument
def optimize(mat=generate_pd(mat_size),initialization=random_initialization(initialization_magnitude),cov_func=lambda x:max_noise/3,normalize_variance = True,verbose = False,plot = False):
    
    # Initialize x values, representing current prediction for minima
    x_values = np.zeros((mat_size,iterations))
    x_values[:,0] = initialization

    curr_iter = 0
    if verbose:
        print_details(curr_iter,x_values)
    loss = np.zeros((iterations))
    loss[curr_iter] = query_function(matrix,x_values[:,curr_iter])[0][0]

    # Iterate over number of iterations
    for curr_iter in range(1,iterations):
        # update represents the change from x_{j} to x_{j+1}
        update = np.zeros((1,mat_size))
        function_evals = np.zeros((num_gradient_calculations,2))
        for i in range(num_gradient_calculations):
            # delta is the random perturbation to x_{j} to calculate approximate gradient
            delta = np.random.rand(mat_size,)
            # calculate the queried values x_{j} +/- nu*delta
            pos_delta_x = x_values[:,(curr_iter-1)]+nu*delta
            neg_delta_x = x_values[:,curr_iter-1]-nu*delta

            # noise_variance is the variance of the noise to the function query
            noise_variance = np.eye(1)*cov_func(curr_iter)
            function_evals[i,0] = query_function(matrix,pos_delta_x,noise_variance)
            function_evals[i,1] = query_function(matrix,neg_delta_x,noise_variance)

            # accumulate the update and multiply by delta
            curr_update = (function_evals[i,0]-function_evals[i,1])*delta
            update = update + curr_update

        function_var = 1

        if normalize_variance:
            # Find variance of function evaluations
            function_eval_mean = np.mean(function_evals)
            function_var = np.sum(np.square(function_evals-function_eval_mean))/(2*num_gradient_calculations)

        # update current 
        update = update.T.reshape(mat_size,1)
        x_values[:,curr_iter] = x_values[:,curr_iter-1]-update.T*alpha/(nu*np.sqrt(function_var)*num_gradient_calculations)
        if verbose:
            if curr_iter % 25 == 0 :
                print_details(curr_iter,x_values)
        loss[curr_iter] = query_function(matrix,x_values[:,curr_iter])[0][0]
    if verbose:
        print_details(curr_iter,x_values)
    if plot:
        plot_surface(matrix,x_values,loss)
    return loss
    

# Plots the quadratic form and the calculated path
# Quadratic form is color coded using coolwarm mapping
# X values color coded using plasma, dark blue representing early and orange/yellow representing later values
def plot_surface(mat,x_values,loss):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
   
    max_x = initialization_magnitude*1.1 #np.abs(np.max(np.max(np.abs(x_values))))*1.2
    x = np.arange(-max_x,max_x,max_x/50)
    y = np.arange(-max_x,max_x,max_x/50)
    X, Y = np.meshgrid(x, y)
    evaluated = np.array([ query_function(mat,np.array([[x],[y]])) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = evaluated.reshape(X.shape)
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False,alpha=0.5)
    lines = ax.scatter(x_values[0,:], x_values[1,:], loss,cmap='plasma',c=np.arange(0,x_values.shape[1]))
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    if normalize_variance:
        normalize_variance_text = "Normalized"
    else:
        normalize_variance_text = "Non-Normalized"
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('{} Convergence Path with {}'.format(normalize_variance_text,noise_func_name))
    plt.savefig("./Surface_Plots/Surface Plot {} noise {} alpha {} iters {} mag {} nu {} maxnoise {} initializations {} calcs {}.png".
        format(normalize_variance_text,noise_func_name,alpha,iterations,initialization_magnitude,nu,max_noise,
            num_initializations,num_gradient_calculations) , bbox_inches='tight',dpi=200)
    if show_plot:
        plt.show()
        plt.close()

# Function for making the printing look nice
# If verbose argument in optimization is true, it will print this
def print_details(curr_iter,x_values):
    x = x_values[0,curr_iter]
    y = x_values[1,curr_iter]
    evaluation = query_function(matrix,x_values[:,curr_iter])[0][0]
    print("Iteration: {0:>5d}{4:3} (x,y): ({1:>6.2f}, {2:>6.2f}) {4:6} Value: {3:6.2f}".format(curr_iter,x,y,evaluation,""))

def parse_args():
    global mat_size,alpha,iterations,initialization_magnitude,nu,max_noise,max_noise,num_initializations
    global num_gradient_calculations,normalize_variance,normalize_variance_text, generate_surface_plots,verbose,show_plot
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat_size", default = Constants.mat_size,action="store",
                    help="matrix size",dest="mat_size",type=int)
    parser.add_argument("--alpha", default = Constants.alpha,action="store",
                    help="learning rate",dest="alpha",type=float)
    parser.add_argument("--iters", default = Constants.iterations,action="store",
                    help="number of iterations",dest="iterations",type=int)
    parser.add_argument("--init_mag", default =Constants.initialization_magnitude,action="store",
                    help="initialization magnitude",dest="initialization_magnitude",type=int)
    parser.add_argument("--nu", default = Constants.nu,action="store",
                    help="standard deviation for query",dest="nu",type=float)
    parser.add_argument("--max_noise", default = Constants.max_noise,action="store",
                    help="maximum noise from oracle",dest="max_noise",type=int)
    parser.add_argument("--num_init", default = Constants.num_initializations,action="store",
                    help="number initializations to process",dest="num_initializations",type=int)
    parser.add_argument("--num_samples", default = Constants.num_gradient_calculations,action="store",
                    help="number of directions to sample to calculate approximate gradient",dest="num_gradient_calculations",type=int)
    parser.add_argument("--no-normalize", default = Constants.normalize_variance,action="store_false",
                    help="boolean to normalize step size",dest="normalize_variance")
    parser.add_argument("--surface_plots", default = False, action="store_true",
                    help="boolean to generate surface plots rather than loss curves",dest="surface_plots")
    parser.add_argument("--verbose", default = False, action="store_true",
                    help="print out the (x,y) location and value",dest="verbose")
    parser.add_argument("--show_plot", default = False, action="store_true",
                    help="show the plot after generation",dest="show_plot")
    args = parser.parse_args()

    mat_size = args.mat_size
    alpha = args.alpha
    iterations = args.iterations
    initialization_magnitude = args.initialization_magnitude
    nu = args.nu
    max_noise = args.max_noise
    num_initializations = args.num_initializations
    num_gradient_calculations = args.num_gradient_calculations
    normalize_variance = args.normalize_variance
    generate_surface_plots = args.surface_plots
    verbose = args.verbose
    show_plot = args.show_plot
    if normalize_variance:
        normalize_variance_text = "Normalized"
    else:
        normalize_variance_text = "Non-Normalized"

if __name__ == '__main__':
    parse_args()
    if generate_surface_plots:
        matrix = generate_pd(mat_size)
        initialization = random_initialization(initialization_magnitude,quad=2)

        noise_func_name = "No Noise"
        optimize(matrix, initialization, cov_func = no_cov,      normalize_variance=normalize_variance,verbose=verbose,plot=True)
        noise_func_name = "Constant Noise"
        optimize(matrix, initialization, cov_func = constant_cov,normalize_variance=normalize_variance,verbose=verbose,plot=True)
        noise_func_name = "Inc. Quad. Noise"
        optimize(matrix, initialization, cov_func = inc_quad_cov,normalize_variance=normalize_variance,verbose=verbose,plot=True)
        noise_func_name = "Dec. Quad. Noise"
        optimize(matrix, initialization, cov_func = dec_quad_cov,normalize_variance=normalize_variance,verbose=verbose,plot=True)

    else:
        loss = np.zeros((iterations,num_initializations,4))
        for trial in range(num_initializations):
            print("Initialization %d of %d"%(trial+1,num_initializations))
            matrix = generate_pd(mat_size)
            initialization = random_initialization(initialization_magnitude)
            #print(initialization,np.linalg.norm(initialization))
            loss0 = optimize(matrix, initialization, cov_func = no_cov,normalize_variance=normalize_variance,verbose=verbose)
            loss1 = optimize(matrix, initialization, cov_func = constant_cov,normalize_variance=normalize_variance,verbose=verbose)
            loss2 = optimize(matrix, initialization, cov_func = inc_quad_cov,normalize_variance=normalize_variance,verbose=verbose)
            loss3 = optimize(matrix, initialization, cov_func = dec_quad_cov,normalize_variance=normalize_variance,verbose=verbose)
            loss[:,trial,0] = loss0
            loss[:,trial,1] = loss1
            loss[:,trial,2] = loss2
            loss[:,trial,3] = loss3

        average_loss = np.mean(loss,axis=1)

        plt.plot(average_loss[:,0])
        plt.plot(average_loss[:,1])
        plt.plot(average_loss[:,2])
        plt.plot(average_loss[:,3])
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.ylim(ymin=-5)
        plt.title('Convergence Rates for Random Search with {} Step Sizes'.format(normalize_variance_text))
        plt.legend(['No Noise','Constant Noise', 'Quad. Inc. Noise', 'Quad. Dec. Noise'], loc='upper right')
        if mat_size == 2:
            plt.savefig("./Loss_Curves/{} alpha {} iters {} mag {} nu {} maxnoise {} initializations {} calcs {}.png".
                format(normalize_variance_text,alpha,iterations,initialization_magnitude,nu,max_noise,num_initializations,num_gradient_calculations) , bbox_inches='tight',dpi=200)
        else:
            plt.savefig("./Loss_Curves/{} matrix size {} alpha {} iters {} mag {} nu {} maxnoise {} initializations {} calcs {}.png".
                format(normalize_variance_text,mat_size,alpha,iterations,initialization_magnitude,nu,max_noise,num_initializations,num_gradient_calculations) , bbox_inches='tight',dpi=200)
        if show_plot:
            plt.show()
    


    