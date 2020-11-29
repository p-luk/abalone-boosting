#!/usr/bin/env python
# coding: utf-8

# In[184]:


import statistics
import matplotlib.pyplot as plt
import numpy as np

class Boost:
    def __init__(self, X_train, y_train, X_test, y_test, T, loss):
        self.T = T
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.X_test = np.array(X_test)
        self.y_test = np.array(y_test)
        self.loss = loss
        self.alphas = None
        self.base_clfs = None
        self.accuracy = []
        self.predictions = None
    
    def fit(self):
        # at t=0, uniform distribution
        d_t = np.ones(len(self.X_train)) / len(self.X_train)
        
        # Run the boosting algorithm by creating T "weighted models"
        alphas = [] 
        base_clfs = []
        
        for t in range(self.T):
            # fit base classifier (decision stump) with weights d_t
            stump = DecisionTreeClassifier(max_depth = 1)
            base_clf = stump.fit(self.X_train, self.y_train, sample_weight = d_t) 
            # add base classifier to list for boosting
            base_clfs.append(base_clf)
            pred = base_clf.predict(self.X_train)
            score = base_clf.score(self.X_train, self.y_train)
            evaluation = np.where(pred == self.y_train,1,0)
            errors = np.where(pred != self.y_train,1,0)
            
            # calculate misclassification rate and accuracy
            accuracy = sum(evaluation) / len(evaluation)
            misclassification = sum(errors) / len(errors)
            # calculate error weighted by distribution
            d_t, alpha = self.loss(d_t, errors)
            alphas.append(alpha)
        
        self.alphas = alphas
        self.base_clfs = base_clfs
            
    def predict(self):
        accuracy = []
        predictions = []
        for alpha, base_clf in zip(self.alphas,self.base_clfs):
            prediction = alpha*base_clf.predict(self.X_test)
            predictions.append(prediction)
            self.accuracy.append(np.sum(np.sign(np.sum(np.array(predictions),axis=0)) == self.y_test) / len(predictions[0]))

def exploss(d_t, errors):
    """
    Returns distribution update and alpha_t using coordinate descent.
    Formula: exp(- alpha_t * y_i * h_t(x_i))
    ===
    Keyword arguments:
    d_t -- np.array-like distribution weights
    errors -- np.array-like list indicating misclassification errors 
    """
    err_t = np.dot(d_t, errors) / np.sum(d_t)
    EPS = 1e-10 # prevent undefined values when err_t = 0
    alpha_t = 0.5 * np.log((1 - err_t + EPS) / (float(err_t) + EPS))
    Z_t = 2 * np.sqrt(err_t * (1. - err_t))
    errors_weighted = [x if x==1 else -1 for x in errors]
    d_next = np.multiply(d_t, np.exp([float(x) * alpha_t for x in errors_weighted]))
    d_next = d_next / Z_t #np.sum(d_next)
    return d_next, alpha_t


def logisticloss(d_t, errors):
    """
    Returns distribution update using coordinate descent.
    Formula: log_2(1 + exp(- alpha_t * y_i * h_t(x_i)))
    ===
    Keyword arguments:
    d -- np.array-like distribution weights
    errors -- np.array-like list indicating misclassification errors 
    """
    err_t = np.dot(d_t, errors) / np.sum(d_t)
    EPS = 1e-10 # prevent undefined values when err_t = 0
    alpha_t = 0.5 * np.log((1 - err_t + EPS) / (float(err_t) + EPS))
    Z_t = 1/np.log(2) * 2 * np.sqrt(err_t * (1. - err_t))
    errors_weighted = [x if x==1 else -1 for x in errors]
    exp = np.exp([- float(x) * alpha_t for x in errors_weighted])
    d_next = np.multiply(d_t, 1 / (np.log(2) * (1 + exp)))
    d_next = d_next / np.sum(d_next)
    return d_next, alpha_t


def cross_validate(train, T_range, loss, nfolds = 10):
    """
    Returns best T using cross-validation over folds.
    ===
    Keyword arguments:
    train -- np.array-like list of input features for train dataset
    T_range -- list of integers for total number of iterations
    loss -- loss function; implemented for 'exploss' (original adaboost) and 'logisticloss'
    nfolds -- number of subsamples per T
    """
    # split training data into folds
    folds = KFold(n_splits=nfolds, shuffle=True)
    accuracies = [None] * len(T_range)
    
    # choose T
    for i, T in enumerate(T_range):
        T_accuracies = [None] * nfolds
        
        # cross-validate over 10 folds per T (effective sample = 10)
        for j, (rest, fold_inds) in enumerate(list(folds.split(train))):
            fold_data = train[fold_inds]
            # split features and labels
            train_fold, dev_fold = train_test_split(fold_data, train_size=0.75, shuffle=True)
            X_train, y_train = train_fold[:, :-1], train_fold[:, -1]
            X_dev, y_dev = dev_fold[:, :-1], dev_fold[:, -1]
            # convert labels to binary classification
            y_train = [1 if i <= 9 else -1 for i in y_train]
            y_dev = [1 if i <= 9 else -1 for i in y_dev]

            # boost on fold
            model = Boost(X_train, y_train, X_dev, y_dev, T, loss)
            model.fit()
            model.predict()
            print('T = {} Fold = {} Accuracy = {}'.format(T, i, model.accuracy[-1]))
            
            # add to accuracies
            T_accuracies[j] = model.accuracy[-1]
        accuracies[i] = T_accuracies
    return accuracies

    

def plot_cv(T_range, accuracies, losslabel):
    """Plots accuracies +/- 1 standard deviation from cross-validation."""
    mean, stdup, stddown = [None] * len(accuracies), [None] * len(accuracies), [None] * len(accuracies)
    for i, cv_acc in enumerate(accuracies):
        mean[i] = statistics.mean(cv_acc)
        std = statistics.stdev(cv_acc)
        stdup[i] = mean[i] + std
        stddown[i] = mean[i] - std
    # plot
    plt.plot(np.array(T_range), np.array(mean), label='Mean')
    plt.plot(np.array(T_range), np.array(stdup), label='+1 std')
    plt.plot(np.array(T_range), np.array(stddown), label='-1 std')
    plt.title('Average Cross-Validation Accuracy Using {}'.format(losslabel))
    plt.xlabel('T')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid()

def main():
    train, test = preprocess_data('./abalone.data')
    # cross-validate over range of T
    T_range = [i*100 for i in range(1,11)]
    cv_exploss = cross_validate(train, T_range, exploss)
    cv_logisticloss = cross_validate(train, T_range, logisticloss)
    # plot results
    plot_cv(T_range, cv_exploss, 'ExpLoss (Original)')
    plot_cv(T_range, cv_logisticloss, 'LogisticLoss')
    
    # retrain models using full train set for optimal T
    # split features and labels
    X_train, y_train = train[:, :-1], train[:, -1]
    X_test, y_test = test[:, :-1], test[:, -1]
    # convert y_test to binary classification
    y_test = [1 if i <= 9 else -1 for i in y_test]
    y_train = [1 if i <= 9 else -1 for i in y_train]
    # compare results
    # logisticloss
    T_logisticloss = 400
    logisticloss_model = Boost(X_train, y_train, X_test, y_test, T_logisticloss, logisticloss)
    logisticloss_model.fit()
    logisticloss_model.predict()

    logisticloss_train = Boost(X_train, y_train, X_train, y_train, T_logisticloss, logisticloss)
    logisticloss_train.fit()
    logisticloss_train.predict()

    plt.plot(range(T_logisticloss), logisticloss_train.accuracy, label='Train')
    plt.plot(range(T_logisticloss), logisticloss_model.accuracy, label='Test')
    plt.title('Accuracy using LogisticLoss \nT={0}, Accuracy={1:.4f}'.format(T_logisticloss, logisticloss_model.accuracy[-1]))
    plt.xlabel('T')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid()
    
    # exploss
    T_exploss=100
    exploss_model = Boost(X_train, y_train, X_test, y_test, T_exploss, exploss)
    exploss_model.fit()
    exploss_model.predict()

    exploss_train = Boost(X_train, y_train, X_train, y_train, T_exploss, exploss)
    exploss_train.fit()
    exploss_train.predict()

    plt.plot(range(T_exploss), exploss_train.accuracy, label='Train')
    plt.plot(range(T_exploss), exploss_model.accuracy, label='Test')
    plt.title('Accuracy using ExpLoss \nT={0}, Accuracy={1:.4f}'.format(T_exploss, exploss_model.accuracy[-1]))
    plt.xlabel('T')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid()


if __name__ == "__main__":
    main()

