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
        mean_std = {}
        for class_i in np.unique(Y):
            idx = np.nonzero(Y[:,0] == class_i)  
            X_i = X[idx]
            
            means = np.mean(X_i, axis=0)
            stds = np.std(X_i, axis=0)
            
            m_s = {}
            for i, (mu, sigma) in enumerate(zip(means, stds)):
                m_s[i] = (mu, sigma)
            
            mean_std[class_i] = m_s
        
        self.mean_std = mean_std

    def gaussian_probability(self, x, mu, sigma):
        exp = np.exp(-0.5 * np.square((x - mu) / sigma))
        return exp / sigma * np.sqrt(2 * np.pi)

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
                    p *= self.gaussian_probability(feature, self.mean_std[class_i][i][0], self.mean_std[class_i][i][1])
                probabilities[class_i] = p
            
            argmax = max(probabilities.keys(), key=(lambda k: probabilities[k]))
            predictions.append(argmax)
        
        return np.reshape(predictions, [numsamples, 1])

# Susy: ~23 error
class LogisticReg(Classifier):
    def __init__(self, parameters = {}):
        self.params = utils.update_dictionary_items({'stepsize': 0.01, 'epochs': 100}, parameters)
        self.weights = None

    def learn(self, X, y):
        pass
    
    def predict(self, Xtest):
        ytest = utils.sigmoid(np.dot(Xtest, self.weights))
        ytest[ytest > 0.5] = 1
        ytest[ytest < 0.5] = 0
        return ytest

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
        pass

    def predict(self,Xtest):
        pass

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
        pass

# Note: high variance in errors! Make sure to run multiple times
# Susy: ~28 error (40 centers)
class KernelLogisticRegression(LogisticReg):
    def __init__(self, parameters = {}):
        self.params = utils.update_dictionary_items({
            'stepsize': 0.01,
            'epochs': 100,
            'centers': 10,
        }, parameters)
        self.weights = None

    def learn(self, X, y):
        pass

    def predict(self, Xtest):
        pass
