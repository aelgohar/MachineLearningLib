import numpy as np
from collections import Counter
import MLCourse.utilities as utils

# Susy: ~50 error
class Classifier:
    def __init__(self, parameters = {}):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}

    def getparams(self):
        return self.params

    def learn(self, Xtrain, ytrain):
        """ Learns using the training data """
        pass

    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        return ytest

# Susy: ~27 error
class LinearRegressionClass(Classifier):
    def __init__(self, parameters = {}):
        self.params = {'regwgt': 0.01}
        self.weights = None

    def learn(self, X, y):
        # Ensure y is {-1,1}
        y = np.copy(y)
        y[y == 0] = -1

        # Dividing by numsamples before adding ridge regularization
        # for additional stability; this also makes the
        # regularization parameter not dependent on numsamples
        # if want regularization disappear with more samples, must pass
        # such a regularization parameter lambda/t
        numsamples = X.shape[0]
        numfeatures = X.shape[1]

        inner = (X.T.dot(X) / numsamples) + self.params['regwgt'] * np.eye(numfeatures)
        self.weights = np.linalg.inv(inner).dot(X.T).dot(y) / numsamples

    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        ytest[ytest > 0] = 1
        ytest[ytest < 0] = 0
        return ytest

# Susy: ~25 error
class NaiveBayes(Classifier):
    def __init__(self, parameters = {}):
        """ Params can contain any useful parameters for the algorithm """
        # Assumes that a bias unit has been added to feature vector as the last feature
        # If usecolumnones is False, it ignores this last feature
        self.params = utils.update_dictionary_items({'usecolumnones': False}, parameters)

    def get_prior(self, ytrain):
        unique, counts = np.unique(ytrain, return_counts=True)
        cnts =  dict(zip(unique, counts))

        self.priors = {k: v/len(ytrain) for k, v in cnts.items()}

    def get_mean_variance(self, X, Y):
        # map of {class : {features: means and standard deviations}}
        mean_std = {}
        for class_i in np.unique(Y):
            # get indices where y == class
            idx = np.nonzero(Y[:,0] == class_i)  
            X_i = X[idx]
            
            # get mean, std of all features
            means = np.mean(X_i, axis=0)
            stds = np.std(X_i, axis=0)
            
            # {feature: (mu, sigma)}
            m_s = {}
            for i, (mu, sigma) in enumerate(zip(means, stds)):
                m_s[i] = (mu, sigma)
            
            mean_std[class_i] = m_s
        
        self.mean_std = mean_std

    def learn(self, Xtrain, ytrain):
        # obtain number of classes
        if ytrain.shape[1] == 1:
            self.numclasses = 2
        else:
            raise Exception('Can only handle binary classification')
        
        if not self.params["usecolumnones"]:
            Xtrain = Xtrain[:, :-1]
            
        self.get_prior(ytrain)
        self.get_mean_variance(Xtrain, ytrain)

    def predict(self, Xtest):
        numsamples = Xtest.shape[0]
        probabilities = {}
        predictions = []

        if not self.params["usecolumnones"]:
            Xtest = Xtest[:, :-1]

        for row in Xtest:
            for class_i, prior in self.priors.items():
                p = prior
                for i, feature in enumerate(row):
                    p *= utils.gaussian_pdf(feature, self.mean_std[class_i][i][0], self.mean_std[class_i][i][1])
                
                probabilities[class_i] = p
                # get class with max probability and append it
                argmax = max(probabilities.keys(), key=(lambda k: probabilities[k]))
            predictions.append(argmax)
        
        # return y's
        return np.reshape(predictions, [numsamples, 1])

# Susy: ~23 error
class LogisticReg(Classifier):
    def __init__(self, parameters = {}):
        self.params = utils.update_dictionary_items({'stepsize': 0.01, 'epochs': 1000}, parameters)
        self.weights = None

    def learn(self, X, y):
        self.weights = np.random.rand(X.shape[1]).reshape(X.shape[1], 1)

        for i in range(self.params['epochs']):
            h = utils.sigmoid(X @ self.weights)
            self.weights = self.weights - self.params['stepsize'] * (X.T @ (h - y))
        
    def predict(self, Xtest):
        return np.round(utils.sigmoid(np.dot(Xtest, self.weights)))

# Susy: ~23 error (4 hidden units)
class NeuralNet(Classifier):
    def __init__(self, parameters={}):
        self.params = utils.update_dictionary_items({
            'nh': 4,
            'transfer': 'sigmoid',
            'stepsize': 0.01,
            'epochs': 10,
        }, parameters)

        if self.params['transfer'] is 'sigmoid':
            self.transfer = utils.sigmoid
            self.dtransfer = utils.dsigmoid
        else:
            # For now, only allowing sigmoid transfer
            raise Exception('NeuralNet -> can only handle sigmoid transfer, must set option transfer to string sigmoid')

        self.wi = None
        self.wo = None

    def learn(self, Xtrain, ytrain):
        numsamples, numfeatures = Xtrain.shape
        nh = self.params['nh']
        
        self.wi = np.random.rand(nh, numfeatures)   # shape: 4, 9
        self.wo = np.random.rand(1, nh) # shape: 1, 4

        for i in range(self.params['epochs']):
            for j in np.random.permutation(numsamples):
                # feed forward
                ah, ao = self.evaluate(Xtrain[j])
                ah = ah.reshape(1, nh)
                ao = ao.reshape(-1, 1)
                x = Xtrain[j].reshape(-1, 1)

                # back prop
                delta2 = np.subtract(ao, ytrain[j])
                update2 = ah.T.dot(delta2)   # shape: 1 x nh
                
                delta1 = np.multiply(delta2.dot(self.wo), self.dsig(ah))
                update1 = x.dot(delta1)
                # d_h = np.multiply(self.wo.T @ oi, np.multiply(ah, 1 - ah)).reshape(-1,1)
                
                # self.wo -=  self.params['stepsize'] * delta2 @ ah.reshape(1, nh)
                # self.wi -= self.params['stepsize'] * np.dot(d_h, Xtrain[j].reshape(1,-1))
                self.wo -=  self.params['stepsize'] * update2.T
                self.wi -= self.params['stepsize'] * update1.T

    def dsig(self,x):
        return np.multiply(x, 1 - x)
        
    def predict(self, Xtest):
        # y = []
        _, ao = self.evaluate(Xtest)
        return np.round(ao)
        # for i in range(len(Xtest)):
        #     # go through rows and add output to y
        #     if ao[i] >= 0.5:
        #         y.append([1])
        #     else:
        #         y.append([0])
        #     # y.append(ao)
        
        # # convert to 0's and 1's and return
        # return y

    def get_cost(self, yhat, y):
        return -1 * (y * np.log(yhat) + (1 - y) * np.log(1 - yhat))

    def dloss(self, yhat, y):
        return -1 * (np.divide(y, yhat) - np.divide(1 - y, 1 - yhat))

    def evaluate(self, inputs):
        # hidden activations
        ah = self.transfer(np.dot(self.wi,inputs.T))

        # output activations
        ao = self.transfer(np.dot(self.wo,ah)).T

        return (
            ah, # shape: [nh, samples]
            ao, # shape: [classes, nh]
        )

    def update(self, inputs, outputs):
        ah, ao = self.evaluate(inputs)
        
        loss = self.loss(ao, outputs)

# Note: high variance in errors! Make sure to run multiple times
# Susy: ~28 error (40 centers)
class KernelLogisticRegression(LogisticReg):
    def __init__(self, parameters = {}):
        self.params = utils.update_dictionary_items({
            'stepsize': 0.01,
            'epochs': 100,
            'centers': 10,
            'kernel': 'linear'
        }, parameters)
        self.weights = None

    def learn(self, X, y):
        kernel = self.params['kernel']
        centers = self.params['centers']
        self.centers = X[:centers]

        Ktrain = np.zeros((len(X), self.params['centers']))
        
        if kernel == 'linear':
            numsamples, numfeatures = X.shape
            for i in range(numsamples):
                for j in range(centers):
                    Ktrain[i][j] = self.linear(self.centers[j], X[i])
        
        elif kernel == 'hamming':
            X = X.reshape(-1, 1)
            numsamples, numfeatures = X.shape
            for i in range(numsamples):
                for j in range(centers):
                    Ktrain[i][j] = self.hamming_distance(self.centers[j], X[i])
        else:
            raise Exception('KernelLogisticRegression -> can only handle linear and hamming kernels')
        
        self.weights = np.random.rand(centers, 1)

        for i in range(self.params['epochs']):
            for j in np.random.permutation(numsamples):
                k = Ktrain[j].reshape(1, -1)
                h = utils.sigmoid(k @ self.weights)
                self.weights = self.weights - self.params['stepsize'] * (k.T @ (h - y[j]))
    
    def predict(self, Xtest):
        kernel = self.params['kernel']
        centers = self.params['centers']
        Ktest = np.zeros((len(Xtest), self.params['centers']))
        
        if kernel == 'linear':
            numsamples, numfeatures = Xtest.shape
            for i in range(numsamples):
                for j in range(centers):
                    Ktest[i][j] = self.linear(self.centers[j], Xtest[i])
        
        elif kernel == 'hamming':
            Xtest = Xtest.reshape(-1, 1)
            numsamples, numfeatures = Xtest.shape
            for i in range(numsamples):
                for j in range(centers):
                    Ktest[i][j] = self.hamming_distance(self.centers[j], Xtest[i])

        return np.round(utils.sigmoid(np.dot(Ktest, self.weights)))

    def linear(self, x1, x2):
        return np.dot(x1, x2)

    def hamming_distance(self, x1, x2):
        return np.count_nonzero(x1 != x2)