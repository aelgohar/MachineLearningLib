import numpy as np
import math
import matplotlib.pyplot as plt
import MLCourse.utilities as utils
from time import time
# -------------
# - Baselines -
# -------------
def l2err(prediction,ytest):
    """ l2 error (i.e., root-mean-squared-error) """
    return np.linalg.norm(np.subtract(prediction, ytest))

def geterror(predictions, ytest):
    # Can change this to other error values
    return l2err(predictions, ytest) / np.sqrt(ytest.shape[0])

class Regressor:
    """
    Generic regression interface; returns random regressor
    Random regressor randomly selects w from a Gaussian distribution
    """
    def __init__(self, parameters = {}):
        self.params = parameters
        self.weights = None

    def getparams(self):
        return self.params

    def learn(self, Xtrain, ytrain):
        # Learns using the traindata
        self.weights = np.random.rand(Xtrain.shape[1])

    def predict(self, Xtest):
        # Most regressors return a dot product for the prediction
        ytest = np.dot(Xtest, self.weights)
        return ytest

class RangePredictor(Regressor):
    """
    Random predictor randomly selects value between max and min in training set.
    """
    def __init__(self, parameters = {}):
        self.params = parameters
        self.min = 0
        self.max = 1

    def learn(self, Xtrain, ytrain):
        # Learns using the traindata
        self.min = np.amin(ytrain)
        self.max = np.amax(ytrain)

    def predict(self, Xtest):
        ytest = np.random.rand(Xtest.shape[0])*(self.max-self.min) + self.min
        return ytest

class MeanPredictor(Regressor):
    """
    Returns the average target value observed; a reasonable baseline
    """
    def __init__(self, parameters = {}):
        self.params = parameters
        self.mean = None

    def learn(self, Xtrain, ytrain):
        # Learns using the traindata
        self.mean = np.mean(ytrain)

    def predict(self, Xtest):
        return np.ones((Xtest.shape[0],))*self.mean

class FSLinearRegression(Regressor):
    """
    Linear Regression with feature selection, and ridge regularization
    """
    def __init__(self, parameters = {}):
        self.params = utils.update_dictionary_items({
            'regwgt': 0.0,
            'features': [1,2,3,4,5],
        }, parameters)

    def learn(self, Xtrain, ytrain):
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:, self.params['features']]
        numfeatures = Xless.shape[1]

        inner = (Xless.T.dot(Xless) / numsamples) + self.params['regwgt'] * np.eye(numfeatures)
        self.weights = np.linalg.inv(inner).dot(Xless.T).dot(ytrain) / numsamples

    def predict(self, Xtest):
        Xless = Xtest[:, self.params['features']]
        ytest = np.dot(Xless, self.weights)
        return ytest

class RidgeLinearRegression(Regressor):
    """
    Linear Regression with ridge regularization (l2 regularization)
    """
    def __init__(self, parameters = {}):
        # Default parameters, any of which can be overwritten by values passed to params
        self.params = utils.update_dictionary_items({'regwgt': 0.01, 'features': range(5)}, parameters)

    def learn(self, Xtrain, ytrain):
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:, self.params['features']]
        numfeatures = Xless.shape[1]

        inner = (Xless.T.dot(Xless) / numsamples) + self.params['regwgt'] * np.eye(numfeatures)
        self.weights = np.linalg.inv(inner).dot(Xless.T).dot(ytrain) / numsamples

    def predict(self, Xtest):
        Xless = Xtest[:, self.params['features']]
        ytest = np.dot(Xless, self.weights)
        return ytest

class LassoRegression(Regressor):
    """
    Lasso Linear Regression
    """
    def __init__(self, parameters = {}):
        # Default parameters, any of which can be overwritten by values passed to params
        self.params = utils.update_dictionary_items({
            'regwgt': 0.5
        }, parameters)
        
        self.weights = None

    def prox(self, w, step_size, regwgt):
        const = step_size * regwgt
        for i in range(w.shape[0]):
            if w[i] > const: 
                w[i] -= const
            elif np.abs(w[i]) <= const: 
                w[i] = 0 
            elif w[i] < -1 * const: 
                w[i] += const
        return w

    def c(self, X, w, y, regwgt):
        # return np.linalg.norm(np.dot(X, w) - y) ** 2 + regwgt * np.linalg.norm(w, ord=1)
        return np.dot((np.dot(X, w) - y).T, np.dot(X, w) - y) + regwgt * np.linalg.norm(w, ord=1)
    
    def learn(self, Xtrain, ytrain):
        numsamples, numfeatures = Xtrain.shape

        w = np.zeros(numfeatures)
        err = float('inf')
        tolerance = 10e-4
        # precompute the matrices
        XX = np.dot(Xtrain.T, Xtrain) / numsamples
        Xy = np.dot(Xtrain.T, ytrain) / numsamples
        # step size
        step_size = 1 / (2 * np.linalg.norm(XX, 'fro'))
        c_w = self.c(Xtrain, w, ytrain, self.params['regwgt']) 
        # ?np.linalg.norm(np.subtract(Xtrain.dot(w), ytrain)) / (2 * numsamples)
        i = 0
        while np.abs(c_w - err) > tolerance:
            err = c_w
            w_new = w - step_size * np.dot(XX, w) + step_size * Xy
            w = self.prox(w_new, step_size, self.params['regwgt'])
            c_w = self.c(Xtrain, w, ytrain, self.params['regwgt']) 

        self.weights = w

    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        return ytest

class SGDLinearRegression(Regressor):
    def __init__(self, parameters = {}):
        # Default parameters, any of which can be overwritten by values passed to params
        self.params = utils.update_dictionary_items({
            'regwgt': 0.01,
            'features': range(385),
            'epochs': 1000,
            'step_size': 0.01
        }, parameters)
        
        self.weights = None

    def learn(self, Xtrain, ytrain):
        numsamples, numfeatures = Xtrain.shape
        
        w = np.random.rand(numfeatures)
        y_axis_cost = [geterror(np.dot(Xtrain,w), ytrain)]
        x_axis_time = [0]
        start_time = time()
        
        for i in range(self.params['epochs']):
            for j in np.random.permutation(numsamples):
                g = np.dot(np.dot(Xtrain[j].T, w) - ytrain[j], Xtrain[j])
                w = w - self.params['step_size'] * g
            y_axis_cost.append(geterror(np.dot(Xtrain,w), ytrain))
            x_axis_time.append(time() - start_time)
        
        #NOTE: If you want to display the graphs, uncomment whichever one you're interested in and also the plt.show()
        # plots cost vs epoch
        # plt.title("Stochastic Gradient Descent Cost vs Epoch", fontsize=19)
        # plt.plot(np.arange(len(y_axis)), y_axis_cost)
        
        # plots cost vs time
        # plt.title("Stochastic Gradient Descent Cost vs Time", fontsize=19)
        # plt.plot(x_axis_time, y_axis_cost)

        # plt.show()
        self.weights = w
    
    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        return ytest

class BGDLinearRegression(Regressor):
    def __init__(self, parameters = {}):
        # Default parameters, any of which can be overwritten by values passed to params
        self.params = utils.update_dictionary_items({
            'regwgt': 0.01,
            'features': [1,2,3,4,5],
            'step_size': 0.01
        }, parameters)
        
        self.weights = None

    def line_search(self, w_t, g):
        n_max, t, tolerance = 1.0, 0.7, 10e-4

        n = n_max
        w = w_t
        iteration, max_iter  = 0, 2000
        obj = self.c(w)

        while iteration < max_iter:
            w = w_t - n * g
            if self.c(w) < obj - tolerance:
                break
            n *= t
            iteration += 1

        return n

    def c(self, w):
        return np.linalg.norm(np.subtract(self.X.dot(w), self.y)) ** 2 / (2 * self.numsamples)
    
    def learn(self, Xtrain, ytrain):
        numsamples, numfeatures = Xtrain.shape
        self.X = Xtrain
        self.y = ytrain
        self.numsamples = numsamples
        regwgt = self.params['regwgt']

        w = np.random.rand(numfeatures)
        err, tolerance, max_iter = float('inf'), 10e-4, 10e5
        iteration = 0
        c_w = self.c(w)
        y_axis_cost = []
        x_axis_time = []
        start_time = time()
        while np.abs(c_w - err) > tolerance and iteration < max_iter:
            err = c_w
            y_axis_cost.append(err)
            x_axis_time.append(time() - start_time)
            g = Xtrain.T.dot(Xtrain.dot(w)-ytrain) / numsamples
            step_size = self.line_search(w, g)
            w = w - step_size * g
            c_w = self.c(w)
            iteration += 1
        print('Batch Gradient Descent took a total of ' + str(iteration) + " iterations.")
        
        # displays cost vs epoch
        # plt.title("Batch Gradient Descent Cost vs Epoch")
        # plt.plot(np.arange(len(y_axis_cost)), y_axis_cost)

        # displays cost vs time
        # plt.title("Batch Gradient Descent Cost vs Time")
        # plt.plot(x_axis_time, y_axis_cost)
        
        # plt.show()
        self.weights = w
    
    def predict(self, Xtest):
        # Xless = Xtest[:, self.params['features']]
        ytest = np.dot(Xtest, self.weights)
        return ytest