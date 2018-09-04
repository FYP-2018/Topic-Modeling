import numpy as np
from scipy.sparse import vstack
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import learning_curve
from sklearn.model_selection import StratifiedKFold

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

# n_crossvali = 4
# datasets_dir = []
# datasets = []
# targets = [None] * n_crossvali


# prepare dataset directory
# for i in range(n_crossvali):
    # datasets_dir.append('../data/20newsgroup/processed_20newsgroup' + str(i) + '.pickle')
    # datasets.append(pickle.load(open('../data/20newsgroup/processed_20newsgroup_' + str(i) + '.pickle', 'r')))
    # targets[i] = pickle.load(open('../data/20newsgroup/20newsgroup_target' + str(i) + '.pickle', 'r'))


n_component = 15
max_iter = 100
valid_iter = 10
n_splits = 3
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
'''
dataset = pickle.load(open('../data/20newsgroup/processed_20newsgroup.pickle', 'r'))
targets = pickle.load(open('../data/20newsgroup/20newsgroup_target.pickle', 'r'))

lda_model = LatentDirichletAllocation(n_components=n_component, learning_method='online',
                                      max_iter=max_iter, random_state=0, learning_offset=50.)


skf = StratifiedKFold(n_splits=n_splits)
splited_index = list(skf.split(X=dataset, y=targets))   # skf.split returns a generator!
train_scores = []        # size: (max_iter / valid_iter) * (n_splits)
test_scores = []        # size: (max_iter / valid_iter) * (n_splits)
train_perplexities = []  # size: (max_iter / valid_iter) * (n_splits)
test_perplexities = []  # size: (max_iter / valid_iter) * (n_splits)


for i in range(int(max_iter / valid_iter)):

    train_s = []
    test_s = []
    train_p = []
    test_p = []

    print '\ntraining ', i * valid_iter + 1, '-th iteration'

    for train_index, test_index in splited_index:
        train_data, test_data = dataset[train_index], dataset[test_index]
        lda_model.partial_fit(train_data)

        train_s.append(lda_model.score(train_data))
        test_s.append(lda_model.score(test_data))

        train_p.append(lda_model.perplexity(train_data))
        test_p.append(lda_model.perplexity(test_data))

    train_scores.append(train_s)
    test_scores.append(test_s)
    train_perplexities.append(train_p)
    test_perplexities.append(test_p)

    print "train_s: ", train_scores[i], " test_s: ", test_scores[i], " train_p: ", train_perplexities[i], " test_p: ", test_perplexities[i]


d = {"max_iter": max_iter, "valid_iter": valid_iter,
     "train_s": train_scores, "test_s": test_scores,
     "train_p": train_perplexities, "test_p": test_perplexities}
pickle.dump(d, open("./cross_validation_score.pickle", 'w'))
print "\n", d
'''

d = pickle.load(open("./cross_validation_score.pickle", 'r'))

from pprint import pprint
pprint(d)

train_scores = np.array(d['train_s'])
test_scores = np.array(d['test_s'])
train_perplexities = np.array(d['train_p'])
test_perplexities = np.array(d['test_p'])

title = "Learning curve of LDA"
start_time = time()
print "start plotting..."

x_axis = range(0, max_iter, valid_iter)
plt.figure(figsize=(12, 8))

ax_1 = plt.subplot(211)
ax_1.fill_between(x_axis, train_scores.min(axis=1), train_scores.max(axis=1), alpha=0.1, color="b")
ax_1.fill_between(x_axis, test_scores.min(axis=1), test_scores.max(axis=1), alpha=0.1, color="r")
for i in range(n_splits):
    ax_1.scatter(x_axis, train_scores[:, i], c='b', marker='+')
    ax_1.scatter(x_axis, test_scores[:, i], c='r', marker='+')

train_score_means = np.mean(train_scores, axis=1)
test_scores_means = np.mean(test_scores, axis=1)

ax_1.plot(x_axis, train_score_means, label='train_scores', c='b')
ax_1.plot(x_axis, test_scores_means, label='test_scores', c='r')

for i, x_coord in enumerate(x_axis):
    ax_1.annotate('{0:.2f}'.format(train_score_means[i]), (x_coord - 2, train_score_means[i] + 200000))
    ax_1.annotate('{0:.2f}'.format(test_scores_means[i]), (x_coord - 2, test_scores_means[i] + 200000))
ax_1.legend(loc="best")


ax_2 = plt.subplot(212)
ax_2.fill_between(x_axis, train_perplexities.min(axis=1), train_perplexities.max(axis=1), alpha=0.1, color="b")
ax_2.fill_between(x_axis, test_perplexities.min(axis=1), test_perplexities.max(axis=1), alpha=0.1, color="r")
for i in range(n_splits):
    ax_2.scatter(x_axis, train_perplexities[:, i], c='b', marker='+')
    ax_2.scatter(x_axis, test_perplexities[:, i], c='r', marker='+')

train_perplexities_means = np.mean(train_perplexities, axis=1)
test_perplexities_means = np.mean(test_perplexities, axis=1)

ax_2.plot(x_axis, train_perplexities_means, label='train_perplexities', c='b')
ax_2.plot(x_axis, test_perplexities_means, label='test_perplexities', c='r')

for i, x_coord in enumerate(x_axis):
    ax_2.annotate('{0:.2f}'.format(train_perplexities_means[i]), (x_coord - 2, train_perplexities_means[i] + 50))
    ax_2.annotate('{0:.2f}'.format(test_perplexities_means[i]), (x_coord - 2, test_perplexities_means[i] + 50))
ax_2.legend(loc="best")

plt.savefig('converge_exploration_full.png')

print "duration: ", time() - start_time
