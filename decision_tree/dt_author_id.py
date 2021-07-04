#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import tree
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
clf = tree.DecisionTreeClassifier(min_samples_split=40)

t0 = time()
clf.fit(features_train, labels_train)
print "training time: ", round(time()-t0,3),"s"


t0 = time()
pred = clf.predict(features_test)
print "predict time: ", round(time()-t0,3),"s"


print "Accuracy score: ", accuracy_score(labels_test, pred)

print "Number of features: ", len(features_train[0])
#########################################################

# Data with min_samples_split=40, features = 10%
# training time:  62.058 s
# predict time:  0.033 s
# 0.978384527873


# Data with min_samples_split=40, features = 10%
# training time:  3.944 s
# predict time:  0.002 s
# Accuracy score:  0.967007963595
# Number of features:  379