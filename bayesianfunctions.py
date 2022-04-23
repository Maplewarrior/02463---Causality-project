# This code was written byKristoffer Hougaard Madsen

import numpy as np
import torch
from torch import nn, distributions
from scipy.spatial.distance import cdist
from scipy.stats import norm
import matplotlib.pyplot as plt

torch.set_default_tensor_type(torch.FloatTensor)

def squared_exponential_kernel(x, y, lengthscale, variance):
    '''
    Function that computes the covariance matrix using a squared-exponential kernel
    '''
    # pair-wise distances, size: NxM
    sqdist = cdist(x.reshape((-1, 1)), y.reshape((-1, 1)), 'sqeuclidean')
    # compute the kernel
    cov_matrix = variance * np.exp(-0.5 * sqdist * (1/lengthscale**2))  # NxM
    return cov_matrix


def fit_predictive_GP(X, y, Xtest, lengthscale, kernel_variance, noise_variance):
    '''
    Function that fit the Gaussian Process. It returns the predictive mean function and
    the predictive covariance function. It follows step by step the algorithm on the lecture
    notes
    '''
    X = np.array(X).reshape(-1, 1)
    y = np.array(y)
    K = squared_exponential_kernel(X, X, lengthscale, kernel_variance)
    L = np.linalg.cholesky(K + noise_variance * np.eye(len(X)))

    # compute the mean at our test points.
    Ks = squared_exponential_kernel(X, Xtest, lengthscale, kernel_variance)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))  #
    mu = Ks.T @ alpha

    v = np.linalg.solve(L, Ks)
    # compute the variance at our test points.
    Kss = squared_exponential_kernel(Xtest, Xtest, lengthscale, kernel_variance)
    covariance = Kss - (v.T @ v)
    
    return mu, covariance


# I am using PyTorch to define the optimization function, it can be done in different other ways.
# It is not the best way to implement it, I suppose
def optimize_GP_hyperparams(Xtrain, ytrain, optimization_steps, learning_rate, mean_prior, prior_std):
    '''
    Methods that run the otpimization of the hyperparams of our GP. We will use
    Gradient Descent because it takes to much time to run grid search at each step
    of bayesian optimization. We use a different definition of the kernel to make the
    optimization more stable

    :param X: training set points
    :param y: training targets
    :return: values for lengthscale, output_var, noise_var that maximize the log-likelihood
    '''
    
    # we are re-defining the kernel because we need it in PyTorch
    def squared_exponential_kernel_torch(x, y, _lambda, variance):
        x = x.squeeze(1).expand(x.size(0), y.size(0))
        y = y.squeeze(0).expand(x.size(0), y.size(0))
        sqdist = torch.pow(x - y, 2)
        k = variance * torch.exp(-0.5 * sqdist * (1/_lambda**2))  # NxM
        return k

    X = np.array(Xtrain).reshape(-1,1)
    y = np.array(ytrain).reshape(-1,1)
    N = len(X)

    # tranform our training set in Tensor
    Xtrain_tensor = torch.from_numpy(X).float()
    ytrain_tensor = torch.from_numpy(y ).float()
    # we should define our hyperparameters as torch parameters where we keep track of
    # the operations to get hte gradients from them
    _lambda = nn.Parameter(torch.tensor(1.), requires_grad=True)
    output_variance = nn.Parameter(torch.tensor(1.), requires_grad=True)
    noise_variance = nn.Parameter(torch.tensor(.5), requires_grad=True)

    # we use Adam as optimizer
    optim = torch.optim.Adam([_lambda, output_variance, noise_variance], lr=learning_rate)

    # optimization loop using the log-likelihood that involves the cholesky decomposition 
    nlls = []
    lambdas = []
    output_variances = []
    noise_variances = []
    iterations = optimization_steps
    for i in range(iterations):
        assert noise_variance >= 0, f"ouch! {i, noise_variance}"
        optim.zero_grad()
        K = squared_exponential_kernel_torch(Xtrain_tensor, Xtrain_tensor, _lambda,
                                                output_variance) + noise_variance * torch.eye(N)
        
        L = torch.cholesky(K)
        _alpha_temp, _ = torch.solve(ytrain_tensor, L)
        _alpha, _ = torch.solve(_alpha_temp, L.t())
        nll = N / 2 * torch.log(torch.tensor(2 * np.pi)) + 0.5 * torch.matmul(ytrain_tensor.transpose(0, 1),
                                                                              _alpha) + torch.sum(torch.log(torch.diag(L)))

        # we have to add the log-likelihood of the prior
        norm = distributions.Normal(loc=mean_prior, scale=prior_std)
        prior_negloglike =  torch.log(_lambda) - norm.log_prob(_lambda)

        nll += 0.9 * prior_negloglike
        nll.backward()

        nlls.append(nll.item())
        lambdas.append(_lambda.item())
        output_variances.append(output_variance.item())
        noise_variances.append(noise_variance.item())
        optim.step()

        # projected in the constraints (lengthscale and output variance should be positive)
        for p in [_lambda, output_variance]:
            p.data.clamp_(min=0.0000001)

        noise_variance.data.clamp_(min=1e-5, max= 0.05)

        
    return _lambda.item(), output_variance.item(), noise_variance.item()

# inspiration from code written by Kristoffer Hougaard Madsen
# solution to BayesianOptimization Exercise 3

# this function will plot var2 as a function of var1
def getGP_plot(var1, var2, filename):
    # exclude first column from dataframe (rownumbers)
    df = pd.read_csv('data/'+str(filename)).iloc[:,1:]

    # training set size
    train_size = 10
    # split data into train and test set
    X = np.array(df['A'])
    y = np.array(df['F'])
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

    ## mean and variance of the log-normal distribution for the lengthscale prior 
    prior_mean = 0.
    prior_std = 3.
    
    # bayesian optimization parameters
    optimization_steps = 500
    learning_rate = 5e-3

    # get optimal hyperparameters from bayesian optimization
    lengthscale, output_var, noise_var = optimize_GP_hyperparams(Xtrain, ytrain, optimization_steps, learning_rate, prior_mean, prior_std)

    # predict function from gaussian process
    mu, covariance = fit_predictive_GP(Xtrain, ytrain, Xtest, lengthscale, optimization_steps, learning_rate)
    std = np.sqrt(np.diag(covariance))
    plt.plot(Xtrain, ytrain, 'ro', label='Training points')
    plt.gca().fill_between(Xtest.flat, mu.reshape(-1) - 2 * std, mu.reshape(-1) + 2 * std,  color='lightblue', alpha=0.5, label=r"$2\sigma$")
    plt.plot(Xtest, mu, 'blue', label=r"$\mu$")
    plt.legend()
    plt.show() 


#getGP_plot("B","C","data_50_F-0.5.csv")