#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
# clf = SVC(kernel='linear')
clf = SVC(kernel='rbf', C=10000.0)


# Reduce the training data to 1% of its original to test the seed gain
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]

t0 = time()
clf.fit(features_train, labels_train)
print "training time: ", round(time()-t0,3),"s"

t0 = time()
pred = clf.predict(features_test)
print "predict time: ", round(time()-t0,3),"s"

# Prediction with rbf, C=1000 and dataset = 1%
#print "Prediction[10]: ", pred[10]
#print "Prediction[26]: ", pred[26]
#print "Prediction[50]: ", pred[50]

# Number of exvents predicted to be Chris(1)
# Prediction with rbf, C=1000 and dataset = 100%
print "Prediction of Chris events: ", pred.sum()

print accuracy_score(labels_test, pred)
#########################################################

# With 100% of the training data and linear kernel: 
# training time:  236.576 s
# predict time:  22.174 s
# 0.984072810011

# With 10% of the training data  and linear kernel: 
# training time:  0.193 s
# predict time:  1.8 s
# 0.884527872582

# With 10% of the training data  and rbf kernel: 
# training time:  0.139 s
# predict time:  1.359 s
# 0.616040955631

# With 10% of the training data  and rbf kernel and C = 10000: 
# training time:  0.132 s
# predict time:  1.067 s
# 0.892491467577

# With 100% of the training data  and rbf kernel and C = 10000: 
# training time:  134.789 s
# predict time:  13.576 s
# 0.990898748578