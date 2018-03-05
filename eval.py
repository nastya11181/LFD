#!/usr/bin/python3
# this script evaluates the classification results

import sys
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, f1_score

# input gold classes (id tweet class)
fgold = sys.argv[1]

# subset of the data ('train' or 'test')
subset = sys.argv[2]

# classification task ('gender' or 'age')
task = sys.argv[3]

# input training classes (id tweet class)
ftrain = sys.argv[4]

# input predicted classes (id tweet prediction)
fpred = sys.argv[5]


# retrieve gold labels
Ytest = []
with open(fgold, 'r') as igold:
    for line in igold.readlines()[1:]:
        Ytest.append(line.split()[-1])

# retrieve predicted labels
Yguess = []
with open(fpred, 'r') as ipred:
    for line in ipred.readlines()[1:]:
        Yguess.append(line.split()[-1])

# retrieve training labels
Ytrain = []
with open(ftrain, 'r') as itrain:
    for line in itrain.readlines()[1:]:
        Ytrain.append(line.split()[-1])


# print header
print()
print('---------------------------------------------------')
print('---------------------------------------------------')
print('RESULTS FOR ' + task + ' CLASSIFICATION (' + subset + ')')
print()

# compare the predicted values with the actual values and print out the accuracy score
print("---------------------------------------------------")
print("Accuracy:", accuracy_score(Ytest, Yguess))
print("---------------------------------------------------")

# print out the precision, recall and the F-score in a table (per class)
print("Precision, recall and F-score per class:")
lab = sorted(set(Ytrain + Ytest)) # these are the sorted labels
lab_print = sorted(set(Yguess + Ytest)) # these are the sorted labels
PRF = precision_recall_fscore_support(Ytest, Yguess, average = None, labels=lab)

# we print out the precision, recall and F-score in a readable way
print("{:10s} {:>10s} {:>10s} {:>10s}".format("", "precision", "recall", "F-score"))
for j, l in enumerate(lab):
    print("{0:10s} {1:10f} {2:10f} {3:10f}".format(l, PRF[0][j],PRF[1][j],PRF[2][j]))

print("---------------------------------------------------")
print("Average F-score (macro):", np.mean(PRF[:][2]))
print("---------------------------------------------------")

# print out the confusion matrix
conf = confusion_matrix(Ytest, Yguess, lab)

print("Confusion matrix:")
print("{:>34s}".format("PREDICTION"))

if len(set(lab)) < 2:
	print("{:18s} {:>7s}".format("", lab[0]))
elif task == 'gender':
    print("{:18s} {:>7s} {:>7s}".format("", lab[0], lab[1]))
elif task == 'age':
    print("{:18s} {:>7s} {:>7s} {:>7s} {:>7s}".format("", lab[0], lab[1], lab[2], lab[3]))


if len(set(lab)) > 1 and len(set(lab_print)) > 1:
    lab_print[0]="GOLD    " + lab[0]
    lab_print[1] = "STANDARD    " + lab[1]
    for j, l in enumerate(lab_print):
        if task == 'gender':
            print("{0:>18s} {1:7d} {2:7d}".format(l, conf[j][0],conf[j][1]))
        if task == 'age':
            print("{0:>18s} {1:7d} {2:7d} {3:7d} {4:7d}".format(l, conf[j][0],conf[j][1], conf[j][2], conf[j][3]))
            print()

