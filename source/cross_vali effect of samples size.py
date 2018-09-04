import numpy as np
from scipy.sparse import vstack
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import learning_curve

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

#pickle
try:
    import cPickle as pickle
except:
    import pickle

import os
os.chdir(r'/home/xzhangbx/remote/others/FYP/feature_extraction_large')

from time import time

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

n_crossvali = 4
# datasets_dir = []
datasets = []
# targets = [None] * n_crossvali

'''
# prepare dataset directory
for i in range(n_crossvali):
    # datasets_dir.append('../data/20newsgroup/processed_20newsgroup' + str(i) + '.pickle')
    datasets.append(pickle.load(open('../data/20newsgroup/processed_20newsgroup_' + str(i) + '.pickle', 'r')))
    # targets[i] = pickle.load(open('../data/20newsgroup/20newsgroup_target' + str(i) + '.pickle', 'r'))
'''


n_component = 15
max_iter = 100
"""
# obtain cross entropy:
for i in range(n_crossvali):
    print type(datasets[i])
    train_data = vstack([datasets[j] for j in range(n_crossvali) if j != i])
    test_data = datasets[i]

    # check shape correction
    # print "train_data.shape - ", train_data.shape, " test_data.shape: ", test_data.shape  # (28269, 9018) ; (9423, 9018)

    # check sparsity (all round 55%)
    '''
    data_dense_train = train_data.todense()
    data_dense_test = test_data.todense()

    print ''
    print "Sparcity (train) :", (float((data_dense_train > 0).sum()) / data_dense_train.size) * 100, '%'
    print "Sparcity (test) :", (float((data_dense_test > 0).sum()) / data_dense_test.size) * 100, '%'
    '''

    # train lda with
    lda_model = LatentDirichletAllocation(n_components=n_component, learning_method='online',
                                          max_iter=max_iter, random_state=0, learning_offset=50.)
"""

print "start loading..."
dataset = pickle.load(open('../data/20newsgroup/processed_20newsgroup.pickle', 'r'))

title = "Learning curve of LDA"
lda_model = LatentDirichletAllocation(n_components=n_component, learning_method='online',
                                      max_iter=max_iter, random_state=0, learning_offset=50.)

start_time = time()
print "start training and plotting..."
plot_learning_curve(lda_model, title, dataset, y=None, n_jobs=4)
print "duration: ", time() - start_time

plt.savefig('lda learning curve.png')