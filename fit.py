#!/usr/bin/python3
# this script uses the extracted features to train and optimize models

import re
import sys
import pickle
from pathlib import Path
from random import randint
from collections import Counter

import numpy as np
import numpy.random as npr

from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score

import statistics as stat
from wordfreq import word_frequency

from nltk.tag import pos_tag

from feats import get_features

from scipy import sparse


# input language
lang = sys.argv[1]

# classification task ('gender' or 'age')
task = sys.argv[2]

# input train text file (id features class)
ftrain_txt = sys.argv[3]

# input test text file (id features class)
ftest_txt = sys.argv[4]

# input train tokenized file (id features class)
ftrain_tok = sys.argv[5]

# input test tokenized file (id features class)
ftest_tok = sys.argv[6]

# optimize parameters (boolean)
optimize = bool(int(sys.argv[7]))

# output training text file (id tweet prediction)
ftrain_out = sys.argv[8]

# output test text file (id tweet prediction)
ftest_out = sys.argv[9]

# output truth file (train)
ftruth_train = sys.argv[10]

# output truth file (test)
ftruth_test = sys.argv[11]


# original data
Xtrain = []
Ytrain = []

# development data (0.2) and training data (0.8)
Xtraindev = []
Ytraindev = []
Xdev = []
Ydev = []

# test data
Xtest = []
Ytest = []

# read input classification labels
with open(ftrain_tok, 'r') as ifile:
    for line in ifile.readlines()[1:]:
        Ytrain.append(line.split('\t')[-1].strip())

with open(ftest_tok, 'r') as ifile:
    for line in ifile.readlines()[1:]:
        Ytest.append(line.split('\t')[-1].strip())

# obtain feature vectors
Xtrain, vec = get_features(ftrain_tok, lang, None)
Xtest, _ = get_features(ftest_tok, lang, vec)

print('Classifying ' + ftrain_tok)

# when optimizing parameters partition the data in train and dev (ratio: 0.8 - 0.2)
if optimize:
    dev_inx = list(set([randint(0, Xtrain.shape[0] - 1) for p in range(int(0.2 * Xtrain.shape[0]))]))
    while len(dev_inx) < int(0.2 * Xtrain.shape[0]):
        dev_inx = list(set(dev_inx + [randint(0, Xtrain.shape[0]) - 1]))
    traindev_inx = [x for x in range(Xtrain.shape[0]) if x not in dev_inx]

    Xtraindev = sparse.csr_matrix(np.take(Xtrain, traindev_inx, axis=0))
    Xdev = sparse.csr_matrix(np.take(Xtrain, dev_inx, axis=0))

    for i in traindev_inx:
        Ytraindev.append(Ytrain[i])
    for i in dev_inx:
        Ydev.append(Ytrain[i])

Xtrain = sparse.csr_matrix(Xtrain)
Xtest = sparse.csr_matrix(Xtest)

# parameters of the SVC to try when optimizing (linear and rbf)
max_acc = 0
C_range = [1, 5, 10, 100]
gamma_range = [0.01, 0.1, 1.0, 10.0]

# a priori best models found
if lang == 'english':
    if task == 'gender':
        best_kernel = 'rbf'
        best_C = 100
        best_gamma = 0.1
    else:
        best_kernel = 'rbf'
        best_C = 10
        best_gamma = 0.1

elif lang == 'spanish':
    if task == 'gender':
        best_kernel = 'rbf'
        best_C = 10
        best_gamma = 0.1
    else:
        best_kernel = 'linear'
        best_C = 10
        best_gamma = 0

elif lang == 'italian':
    best_kernel = 'linear'
    best_C = 5
    best_gamma = 0

elif lang == 'dutch':
    best_kernel = 'linear'
    best_C = 5
    best_gamma = 0


# classifiers to use
cls_range = []
if optimize:
    for c_val in C_range:
        cls_range.append(SVC(kernel='linear', C=c_val))
        for gamma_val in gamma_range:
            cls_range.append(SVC(kernel='rbf', gamma=gamma_val, C=c_val))


# validate each classifier using the test set
for cls in cls_range:
    if len(set(Ytrain)) > 1:
        cls.fit(Xtraindev, Ytraindev)
        Ydev_guess = cls.predict(Xdev)
        acc = accuracy_score(Ydev, Ydev_guess)

        if acc > max_acc:
            best_kernel = cls.kernel
            best_C = cls.C
            best_gamma = cls.gamma
            max_acc = acc

        if cls.kernel == 'linear':
            print('Accuracy with linear kernel, c: ' + str(cls.C) + ' is: ', acc)
        else:
            print('Accuracy with rbf kernel, c: ' + str(cls.C) + ', gamma: ' + str(cls.gamma) + ' is: ', acc)
        sys.stdout.flush()

print()
print('! Best parameter values. Kernel: ' + str(best_kernel) + ', c: ' + str(best_C) + ', gamma: ' + str(best_gamma))


# define classifier to use
if best_kernel == 'linear':
    clf = SVC(kernel="linear", C=best_C)
if best_kernel == 'rbf':
    clf = SVC(kernel="rbf", C=best_C, gamma=best_gamma)


# fit the final classifier on the full data and predict test data
if len(set(Ytrain)) > 1:
    clf.fit(Xtrain, Ytrain)
    Ytrain_guess = clf.predict(Xtrain)
    Ytest_guess = clf.predict(Xtest)
else:
    Ytrain_guess = Ytrain
    Ytest_guess = [Ytrain[0]] * int(Xtest.shape[0])


# classification results per author
auth2labels_train = {}
auth2labels_test = {}

# print train results into a file
with open(ftrain_out, 'w') as otrain:
    with open(ftrain_txt, 'r') as itrain:
        lines = itrain.readlines()
        otrain.write('id\ttweet\t' + str(task) + '\n')
        for i in range(len(Ytrain_guess)):
            fields = lines[i+1].split('\t')
            otrain.write(fields[0].strip() + '\t' + fields[1].strip() + '\t' + Ytrain_guess[i] + '\n')

            # accumulate classification results
            if fields[0].strip() not in auth2labels_train:
                auth2labels_train[fields[0].strip()] = []
            auth2labels_train[fields[0].strip()].append(Ytrain_guess[i])


# print test results into a file and collect labels
with open(ftest_out, 'w') as otest:
    with open(ftest_txt, 'r') as itest:
        lines = itest.readlines()
        otest.write('id\ttweet\t' + str(task) + '\n')
        for i in range(len(Ytest_guess)):
            fields = lines[i+1].split('\t')
            otest.write(fields[0].strip() + '\t' + fields[1].strip() + '\t' + Ytest_guess[i] + '\n')

            # accumulate classification results
            if fields[0].strip() not in auth2labels_test:
                auth2labels_test[fields[0].strip()] = []
            auth2labels_test[fields[0].strip()].append(Ytest_guess[i])


# check if truth file exists, else, initialize
if not Path(ftruth_train).is_file():
    with open(ftruth_train, 'w') as otruth:
        for auth in auth2labels_train:
            otruth.write(auth + '\n')

if not Path(ftruth_test).is_file():
    with open(ftruth_test, 'w') as otruth:
        for auth in auth2labels_test:
            otruth.write(auth + '\n')


# output results into the truth file
newlines_train = []
newlines_test = []

with open(ftruth_train, 'r') as otruth:
    for line in otruth.readlines():
        auth = line.split(':::')[0].strip()
        if auth in auth2labels_train:
            label = Counter(auth2labels_train[auth]).most_common(1)[0][0]
            newlines_train.append(line.rstrip('\n') + ':::' + str(label) + '\n')

with open(ftruth_train, 'w') as otruth:
    for line in newlines_train:
        otruth.write(line)

with open(ftruth_test, 'r') as otruth:
    for line in otruth.readlines():
        auth = line.split(':::')[0].strip()
        if auth in auth2labels_test:
            label = Counter(auth2labels_test[auth]).most_common(1)[0][0]
            newlines_test.append(line.rstrip('\n') + ':::' + str(label) + '\n')

with open(ftruth_test, 'w') as otruth:
    for line in newlines_test:
        otruth.write(line)

