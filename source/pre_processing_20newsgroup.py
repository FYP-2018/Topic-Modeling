# sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups

#pickle
try:
    import cPickle as pickle
except:
    import pickle

# others
import re, nltk, gensim
from nltk.stem import PorterStemmer


# system
from pprint import pprint

import os
os.chdir(r'/home/xzhangbx/remote/others/FYP/feature_extraction_large')


def sent_to_words(sentences):
    ''' remove punctuation and unnecessary characters '''
    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))



n_samples = 2000
removes = ('headers', 'footers', 'quotes')

newsgroups_all = fetch_20newsgroups(subset='all', remove=removes, download_if_missing=False)
pprint(newsgroups_all.data[:1])

# preprocessing data:
# remove \n
data = [re.sub('\s+', ' ', sent) for sent in newsgroups_all.data]
# remove single quota
data = [re.sub("\'", ' ', sent) for sent in data]
# remove non-alphabet
data = [re.sub('[^a-zA-Z]+', ' ', sent) for sent in data] # add by myself
# pprint(data[:1])

# tokenize and clean up
data_words = list(sent_to_words(data))
print ''
print(data_words[:1])


# stem
data_lemmatized = []
ps = PorterStemmer()
for doc_i in range(len(data_words)):
    sentence = []
    for word_j in range(len(data_words[doc_i])):
        new_word = ps.stem(data_words[doc_i][word_j])
        if len(new_word) >= 3:
            sentence.append(new_word)

    data_lemmatized.append(' '.join(sentence))

print ''
print data_lemmatized[:1]

foutput = open('../data/20newsgroup/processed_20newsgroup_all_news.txt', 'w')
for article in data_lemmatized:
    foutput.write(' '.join(article))
foutput.close()


# doc-word matrix
vectorizer = CountVectorizer(analyzer='word', min_df=10, stop_words='english', lowercase=True,
                             token_pattern='[a-zA-Z0-9]{3,}')
data_vectorized = vectorizer.fit_transform(data_lemmatized)


# pickle.dump(vectorizer, open('../data/20newsgroup/processed_20newsgroup_model.pickle', 'w'))
# pickle.dump(data_vectorized, open('../data/20newsgroup/processed_20newsgroup.pickle', 'w'))
# pickle.dump(newsgroups_all.target, open('../data/20newsgroup/20newsgroup_target.pickle', 'w'))

print ''
print data_vectorized[:1]