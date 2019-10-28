import csv
import random
import math
import numpy as np

import regressionalgorithms as algs

import MLCourse.dataloader as dtl
import MLCourse.plotfcns as plotfcns

def l2err(prediction,ytest):
    """ l2 error (i.e., root-mean-squared-error) """
    return np.linalg.norm(np.subtract(prediction, ytest))

def l1err(prediction,ytest):
    """ l1 error """
    return np.linalg.norm(np.subtract(prediction, ytest), ord=1)

def l2err_squared(prediction,ytest):
    """ l2 error squared """
    return np.square(np.linalg.norm(np.subtract(prediction, ytest)))

def geterror(predictions, ytest):
    # Can change this to other error values
    return l2err(predictions, ytest) / np.sqrt(ytest.shape[0])


if __name__ == '__main__':
    trainsize = 5000
    testsize = 5000
    numruns = 5

    # choose which models you want to run
    regressionalgs = {
        'Random': algs.Regressor,
        'Mean': algs.MeanPredictor,
        'FSLinearRegression': algs.FSLinearRegression,
        'RidgeLinearRegression': algs.RidgeLinearRegression,
        'LassoRegression': algs.LassoRegression,
        'SGDRegression': algs.SGDLinearRegression,
        'BGDRegression': algs.BGDLinearRegression,
    }
    numalgs = len(regressionalgs)

    # Specify the name of the algorithm and an array of parameter values to try
    # if an algorithm is not include, will run with default parameters
    # NOTE: Added more feature ranges to test a varying number of features including the entire feature set
    parameters = {
        'FSLinearRegression': [
            # { 'features': range(25) },
            { 'features': range(50) },
            # { 'features': [1, 2, 3, 4, 5] },
        ],
        'RidgeLinearRegression': [
            # { 'regwgt': 0.00 },
            { 'features': range(385) },
            # { 'regwgt': 0.05 },
        ],
        'SGDRegression': [
            {'epochs': 1000,
            'step_size': 0.01}
        ]
    }

    errors = {}
    for learnername in regressionalgs:
        # get the parameters to try for this learner
        # if none specified, then default to an array of 1 parameter setting: None
        params = parameters.get(learnername, [ None ])
        errors[learnername] = np.zeros((len(params), numruns))

    for r in range(numruns):
        trainset, testset = dtl.load_ctscan(trainsize,testsize)
        print(('Running on train={0} and test={1} samples for run {2}').format(trainset[0].shape[0], testset[0].shape[0], r))

        for learnername, Learner in regressionalgs.items():
            params = parameters.get(learnername, [ None ])
            for p in range(len(params)):
                learner = Learner(params[p])
                print ('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
                # Train model
                learner.learn(trainset[0], trainset[1])
                # Test model
                predictions = learner.predict(testset[0])
                error = geterror(testset[1], predictions)
                print ('Error for ' + learnername + ': ' + str(error))
                errors[learnername][p, r] = error

    for learnername in regressionalgs:
        params = parameters.get(learnername, [ None ])
        besterror = np.mean(errors[learnername][0, :])
        bestparams = 0
        for p in range(len(params)):
            aveerror = np.mean(errors[learnername][p, :])
            if aveerror < besterror:
                stderror = np.std(errors[learnername][p, :]) / sqrt_num_runs
                besterror = aveerror
                bestparams = p

        # Extract best parameters
        best = params[bestparams]
        print ('Best parameters for ' + learnername + ': ' + str(best))
        print ('Average error for ' + learnername + ': ' + str(besterror) + ' +- ' + str(1.96 * np.std(errors[learnername][bestparams, :]) / math.sqrt(numruns)))
        print ('Standard error for ' + learnername + ': ' + str(np.std(errors[learnername][bestparams, :]) / math.sqrt(numruns)))