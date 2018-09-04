from sklearn.decomposition import LatentDirichletAllocation  # our model
from sklearn.model_selection import StratifiedKFold          # for cross-evaluation

import numpy as np

try:
    import cPickle as pickle    # since cPickle is faster than pickle, use that if possible
except:
    import pickle

import os
from pprint import pprint
from time import time

import matplotlib
matplotlib.use('agg')   # for linux env only (not required for windows
import matplotlib.pyplot as plt

from functions import plot_scores

os.chdir(r'/home/xzhangbx/remote/others/FYP/test with 20newsgroup')     # depending on the path on server


n_component = 11
learning_decays = [0.5, 0.6, 0.7, 0.8, 0.9]

max_iter = 100
valid_iter = 10
n_splits = 3

start_time = time()
dict_learningDecay = {}

# (1) train the model if no score dictionary is found
if not os.path.isfile("./cross_validation_score_learningDecay(n_topic"+str(n_component)+".pickle"):
    print "start loading dataset ..."

    dataset = pickle.load(open('../data/20newsgroup/processed_20newsgroup.pickle', 'r'))
    targets = pickle.load(open('../data/20newsgroup/20newsgroup_target.pickle', 'r'))

    print 'processed 20newsgroup dataset loaded, with dataset shape: ', dataset.shape
    print 'processed 20newsgroup target loaded, with dataset shape: ', targets.shape


    # train lda models with different etaa priors
    for learning_decay in learning_decays:
        print '\n\n ------------------ # LEARNING DECAY =', learning_decay, ' -----------------------'
        lda_model = LatentDirichletAllocation(n_components=n_component, learning_method='online', learning_decay=learning_decay,
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

            print "train_scores: ", train_scores[i], " test_scores: ", test_scores[i], " train_perplexities: ", train_perplexities[i], " test_perplexities: ", test_perplexities[i]


        # store the scores into a single dictionary
        dict_learningDecay[str(learning_decay) + '_learning_decay'] = {
            "max_iter": max_iter, "valid_iter": valid_iter,
            "train_scores": train_scores, "test_scores": test_scores,
            "train_perplexities": train_perplexities, "test_perplexities": test_perplexities
        }

    pickle.dump(dict_learningDecay, open("./cross_validation_score_learningDecay(n_topic"+str(n_component)+".pickle", 'w'))
    pprint(dict_learningDecay)

else:
    dict_learningDecay = pickle.load(open("./cross_validation_score_learningDecay(n_topic"+str(n_component)+".pickle", 'r'))

print "\nFinish Loading/Training within", time() - start_time, 'secends'
start_time = time()

# (2) print out the evaluation scores
print "start plotting..."

color_couples = [('#99c9eb', '#f998a5'), ('#4ca1dd', '#f77687'),
                 ('#0079cf', '#f6546a'),
                 ('#005490', '#c44354'), ('#003052', '#93323f'), ]
x_axis = range(0, max_iter, valid_iter)

plt.figure(figsize=(20, 16))
plt.suptitle('Tuning learning decay (n_topic = '+str(n_component)+')', fontsize=12)
ax_1 = plt.subplot(221)
ax_2 = plt.subplot(222)
ax_3 = plt.subplot(223)
ax_4 = plt.subplot(224)

for index, learning_decay in enumerate(learning_decays):
    (c1, c2) = color_couples[index]

    # prepare the scores array (y-axis)
    d = dict_learningDecay[str(learning_decay) + '_learning_decay']
    train_scores = np.array(d['train_scores'])
    test_scores = np.array(d['test_scores'])
    train_perplexities = np.array(d['train_perplexities'])
    test_perplexities = np.array(d['test_perplexities'])

    alpha = 0.1
    label = 'learning_decay '+str(learning_decay)
    plot_scores(ax_1, x_axis, train_scores, n_splits, c1, title='train_scores', label=label)
    plot_scores(ax_2, x_axis, train_perplexities, n_splits, c1, title='train_perplexities', label=label)
    plot_scores(ax_3, x_axis, test_scores, n_splits, c2, title='test_scores', label=label)
    plot_scores(ax_4, x_axis, test_perplexities, n_splits, c2, title='test_perplexities', label=label)

    plt.savefig('converge_exploration_learning_decay(n_topic'+str(n_component)+')_noRange.png')

    print "\nFinish Plotting within", time() - start_time, 'secends'