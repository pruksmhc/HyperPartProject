# Hyperpartisan news detection- exploratory analyses and model selection- Naive Bayes approaches v0.2
# Author: Yash Deshpande
# Last modified: December 2nd, 2018
# Updates:
# 1. Added all naive Bayes' approaches, including CNB
# 2. Working with fully cleaned data

import datetime

print("Naive Bayes script started at " + str(datetime.datetime.now()))

import pandas as pd 
import string
import numpy as np
from sklearn import model_selection as ms
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn import metrics

# Loading data and mapping character classes to numerics 

training_data_df = pd.read_csv('sample_data_labelled_cleaned.csv.gz', sep = '|', header = 0)

training_data_df['Hyperpartisan'][training_data_df['Hyperpartisan'] == True] = 1
training_data_df['Hyperpartisan'][training_data_df['Hyperpartisan'] == False] = 0

X = training_data_df['Article']
Y = training_data_df['Hyperpartisan']
Y = Y.astype(int)
X_train, X_test, Y_train, Y_test = ms.train_test_split(X, Y, test_size = 0.25)

# Vectorizing inputs using TF-IDF
ti_vectorize_data_12 = TfidfVectorizer(ngram_range=(1, 2), stop_words = 'english').fit(X)
x_train_vec_tfidf_12, x_test_vec_tfidf_12 = ti_vectorize_data_12.transform(X_train), ti_vectorize_data_12.transform(X_test)

ti_vectorize_data_14 = TfidfVectorizer(ngram_range=(1, 4), stop_words = 'english').fit(X)
x_train_vec_tfidf_14, x_test_vec_tfidf_14 = ti_vectorize_data_14.transform(X_train), ti_vectorize_data_14.transform(X_test)

ti_vectorize_data_16 = TfidfVectorizer(ngram_range=(1, 6), stop_words = 'english').fit(X)
x_train_vec_tfidf_16, x_test_vec_tfidf_16 = ti_vectorize_data_16.transform(X_train), ti_vectorize_data_16.transform(X_test)

ti_vectorize_data_610 = TfidfVectorizer(ngram_range=(6, 10), stop_words = 'english').fit(X)
x_train_vec_tfidf_610, x_test_vec_tfidf_610 = ti_vectorize_data_610.transform(X_train), ti_vectorize_data_610.transform(X_test)

# Bernoulli Naive Bayes models
BNB_model_ti_12 =  BernoulliNB().fit(x_train_vec_tfidf_12, Y_train)
preds_BNB_tfidf_12 = BNB_model_ti_12.predict_proba(x_test_vec_tfidf_12)
accuracy_BNB_12 = metrics.accuracy_score(Y_test, BNB_model_ti_12.predict(x_test_vec_tfidf_12))

BNB_model_ti_14 =  BernoulliNB().fit(x_train_vec_tfidf_14, Y_train)
preds_BNB_tfidf_14 = BNB_model_ti_14.predict_proba(x_test_vec_tfidf_14)
accuracy_BNB_14 = metrics.accuracy_score(Y_test, BNB_model_ti_14.predict(x_test_vec_tfidf_14))

BNB_model_ti_16 =  BernoulliNB().fit(x_train_vec_tfidf_16, Y_train)
preds_BNB_tfidf_16 = BNB_model_ti_16.predict_proba(x_test_vec_tfidf_16)
accuracy_BNB_16 = metrics.accuracy_score(Y_test, BNB_model_ti_16.predict(x_test_vec_tfidf_16))

BNB_model_ti_610 =  BernoulliNB().fit(x_train_vec_tfidf_610, Y_train)
preds_BNB_tfidf_610 = BNB_model_ti_610.predict_proba(x_test_vec_tfidf_610)
accuracy_BNB_610 = metrics.accuracy_score(Y_test, BNB_model_ti_610.predict(x_test_vec_tfidf_610))

# Multinomial Naive Bayes models
MNB_model_ti_12 =  MultinomialNB().fit(x_train_vec_tfidf_12, Y_train)
preds_MNB_tfidf_12 = MNB_model_ti_12.predict_proba(x_test_vec_tfidf_12)
accuracy_MNB_12 = metrics.accuracy_score(Y_test, BNB_model_ti_12.predict(x_test_vec_tfidf_12))

MNB_model_ti_14 =  MultinomialNB().fit(x_train_vec_tfidf_14, Y_train)
preds_MNB_tfidf_14 = MNB_model_ti_14.predict_proba(x_test_vec_tfidf_14)
accuracy_MNB_14 = metrics.accuracy_score(Y_test, BNB_model_ti_14.predict(x_test_vec_tfidf_14))

MNB_model_ti_16 =  MultinomialNB().fit(x_train_vec_tfidf_16, Y_train)
preds_MNB_tfidf_16 = MNB_model_ti_16.predict_proba(x_test_vec_tfidf_16)
accuracy_MNB_16 = metrics.accuracy_score(Y_test, BNB_model_ti_16.predict(x_test_vec_tfidf_16))

MNB_model_ti_610 =  MultinomialNB().fit(x_train_vec_tfidf_610, Y_train)
preds_MNB_tfidf_610 = MNB_model_ti_610.predict_proba(x_test_vec_tfidf_610)
accuracy_MNB_610 = metrics.accuracy_score(Y_test, BNB_model_ti_610.predict(x_test_vec_tfidf_610))

# Complement Naive Bayes models
CNB_model_ti_12 =  ComplementNB().fit(x_train_vec_tfidf_12, Y_train)
preds_CNB_tfidf_12 = CNB_model_ti_12.predict_proba(x_test_vec_tfidf_12)
accuracy_CNB_12 = metrics.accuracy_score(Y_test, CNB_model_ti_12.predict(x_test_vec_tfidf_12))

CNB_model_ti_14 =  ComplementNB().fit(x_train_vec_tfidf_14, Y_train)
preds_CNB_tfidf_14 = CNB_model_ti_14.predict_proba(x_test_vec_tfidf_14)
accuracy_CNB_14 = metrics.accuracy_score(Y_test, CNB_model_ti_14.predict(x_test_vec_tfidf_14))

CNB_model_ti_16 =  ComplementNB().fit(x_train_vec_tfidf_16, Y_train)
preds_CNB_tfidf_16 = CNB_model_ti_16.predict_proba(x_test_vec_tfidf_16)
accuracy_CNB_16 = metrics.accuracy_score(Y_test, CNB_model_ti_16.predict(x_test_vec_tfidf_16))

CNB_model_ti_610 =  ComplementNB().fit(x_train_vec_tfidf_610, Y_train)
preds_CNB_tfidf_610 = CNB_model_ti_610.predict_proba(x_test_vec_tfidf_610)
accuracy_CNB_610 = metrics.accuracy_score(Y_test, CNB_model_ti_610.predict(x_test_vec_tfidf_610))

# Accuracies DF
accuracies_list = [['BNB_12', accuracy_BNB_12], 
['BNB_14', accuracy_BNB_14],
['BNB_16', accuracy_BNB_16],
['BNB_610', accuracy_BNB_610],
['MNB_12', accuracy_MNB_12],
['MNB_14', accuracy_MNB_14],
['MNB_16', accuracy_MNB_16],
['MNB_610', accuracy_MNB_610],
['CNB_12', accuracy_CNB_12],
['CNB_14', accuracy_CNB_14],
['CNB_16', accuracy_CNB_16],
['CNB_610', accuracy_CNB_610]]
NB_accuracies_df = pd.DataFrame(accuracies_list)
NB_accuracies_df.columns = ['Model', 'Accuracy']
NB_accuracies_df.to_csv("NB_accuracies_df.csv", sep = '|')

# FPR, TPR calculations for all models
fpr_BNB_ti_12, tpr_BNB_ti_12, thresholds_BNB_ti_12 = metrics.roc_curve(Y_test, preds_BNB_tfidf_12[:,1])
fpr_BNB_ti_14, tpr_BNB_ti_14, thresholds_BNB_ti_14 = metrics.roc_curve(Y_test, preds_BNB_tfidf_14[:,1])
fpr_BNB_ti_16, tpr_BNB_ti_16, thresholds_BNB_ti_16 = metrics.roc_curve(Y_test, preds_BNB_tfidf_16[:,1])
fpr_BNB_ti_610, tpr_BNB_ti_610, thresholds_BNB_ti_610 = metrics.roc_curve(Y_test, preds_BNB_tfidf_610[:,1])

fpr_MNB_ti_12, tpr_MNB_ti_12, thresholds_MNB_ti_12 = metrics.roc_curve(Y_test, preds_MNB_tfidf_12[:,1])
fpr_MNB_ti_14, tpr_MNB_ti_14, thresholds_MNB_ti_14 = metrics.roc_curve(Y_test, preds_MNB_tfidf_14[:,1])
fpr_MNB_ti_16, tpr_MNB_ti_16, thresholds_MNB_ti_16 = metrics.roc_curve(Y_test, preds_MNB_tfidf_16[:,1])
fpr_MNB_ti_610, tpr_MNB_ti_610, thresholds_MNB_ti_610 = metrics.roc_curve(Y_test, preds_MNB_tfidf_610[:,1])

fpr_CNB_ti_12, tpr_CNB_ti_12, thresholds_CNB_ti_12 = metrics.roc_curve(Y_test, preds_CNB_tfidf_12[:,1])
fpr_CNB_ti_14, tpr_CNB_ti_14, thresholds_CNB_ti_14 = metrics.roc_curve(Y_test, preds_CNB_tfidf_14[:,1])
fpr_CNB_ti_16, tpr_CNB_ti_16, thresholds_CNB_ti_16 = metrics.roc_curve(Y_test, preds_CNB_tfidf_16[:,1])
fpr_CNB_ti_610, tpr_CNB_ti_610, thresholds_CNB_ti_610 = metrics.roc_curve(Y_test, preds_CNB_tfidf_610[:,1])

# AUC calculations for all models 
roc_auc_BNB_ti_12 = metrics.auc(fpr_BNB_ti_12, tpr_BNB_ti_12)
roc_auc_BNB_ti_14 = metrics.auc(fpr_BNB_ti_14, tpr_BNB_ti_14)
roc_auc_BNB_ti_16 = metrics.auc(fpr_BNB_ti_16, tpr_BNB_ti_16)
roc_auc_BNB_ti_610 = metrics.auc(fpr_BNB_ti_610, tpr_BNB_ti_610)

roc_auc_MNB_ti_12 = metrics.auc(fpr_MNB_ti_12, tpr_MNB_ti_12)
roc_auc_MNB_ti_14 = metrics.auc(fpr_MNB_ti_14, tpr_MNB_ti_14)
roc_auc_MNB_ti_16 = metrics.auc(fpr_MNB_ti_16, tpr_MNB_ti_16)
roc_auc_MNB_ti_610 = metrics.auc(fpr_MNB_ti_610, tpr_MNB_ti_610)

roc_auc_CNB_ti_12 = metrics.auc(fpr_CNB_ti_12, tpr_CNB_ti_12)
roc_auc_CNB_ti_14 = metrics.auc(fpr_CNB_ti_14, tpr_CNB_ti_14)
roc_auc_CNB_ti_16 = metrics.auc(fpr_CNB_ti_16, tpr_CNB_ti_16)
roc_auc_CNB_ti_610 = metrics.auc(fpr_CNB_ti_610, tpr_CNB_ti_610)

# Plotting ROC curves
# BNB
plt.figure(figsize = (20, 10))
plt.title("ROC curves- Bernoulli Naive Bayes for various n-gram ranges", fontsize = 20)
plt.ylabel("True positive rate (TPR)", size = 15)
plt.xlabel("False positive rate (FPR)", size = 15)

plt.plot(fpr_BNB_ti_12, tpr_BNB_ti_12, 'y', c='r', label = 'n-gram range (1, 2)' +' (AUC = %0.3f)' % roc_auc_BNB_ti_12)
plt.plot(fpr_BNB_ti_14, tpr_BNB_ti_14, 'y', c='g', label = 'n-gram range (1, 4)' +' (AUC = %0.3f)' % roc_auc_BNB_ti_14)
plt.plot(fpr_BNB_ti_16, tpr_BNB_ti_16, 'y', c='b', label = 'n-gram range (1, 6)' +' (AUC = %0.3f)' % roc_auc_BNB_ti_16)
plt.plot(fpr_BNB_ti_610, tpr_BNB_ti_610, 'y', c='k', label = 'n-gram range (6, 10)' +' (AUC = %0.3f)' % roc_auc_BNB_ti_610)

x = np.linspace(0, 1, 5)
plt.plot(x, x, 'k--')

plt.legend(loc = "lower right")
plt.savefig("MS_plot_1_BNB.jpg")

# MNB
plt.figure(figsize = (20, 10))
plt.title("ROC curves- Multinomial Naive Bayes for various n-gram ranges", fontsize = 20)
plt.ylabel("True positive rate (TPR)", size = 15)
plt.xlabel("False positive rate (FPR)", size = 15)

plt.plot(fpr_MNB_ti_12, tpr_MNB_ti_12, 'y', c='r', label = 'n-gram range (1, 2)' +' (AUC = %0.3f)' % roc_auc_MNB_ti_12)
plt.plot(fpr_MNB_ti_14, tpr_MNB_ti_14, 'y', c='g', label = 'n-gram range (1, 4)' +' (AUC = %0.3f)' % roc_auc_MNB_ti_14)
plt.plot(fpr_MNB_ti_16, tpr_MNB_ti_16, 'y', c='b', label = 'n-gram range (1, 6)' +' (AUC = %0.3f)' % roc_auc_MNB_ti_16)
plt.plot(fpr_MNB_ti_610, tpr_MNB_ti_610, 'y', c='k', label = 'n-gram range (6, 10)' +' (AUC = %0.3f)' % roc_auc_MNB_ti_610)

x = np.linspace(0, 1, 5)
plt.plot(x, x, 'k--')

plt.legend(loc = "lower right")
plt.savefig("MS_plot_2_MNB.jpg")

# CNB
plt.figure(figsize = (20, 10))
plt.title("ROC curves- Complement Naive Bayes for various n-gram ranges", fontsize = 20)
plt.ylabel("True positive rate (TPR)", size = 15)
plt.xlabel("False positive rate (FPR)", size = 15)

plt.plot(fpr_CNB_ti_12, tpr_CNB_ti_12, 'y', c='r', label = 'n-gram range (1, 2)' +' (AUC = %0.3f)' % roc_auc_CNB_ti_12)
plt.plot(fpr_CNB_ti_14, tpr_CNB_ti_14, 'y', c='g', label = 'n-gram range (1, 4)' +' (AUC = %0.3f)' % roc_auc_CNB_ti_14)
plt.plot(fpr_CNB_ti_16, tpr_CNB_ti_16, 'y', c='b', label = 'n-gram range (1, 6)' +' (AUC = %0.3f)' % roc_auc_CNB_ti_16)
plt.plot(fpr_CNB_ti_610, tpr_CNB_ti_610, 'y', c='k', label = 'n-gram range (6, 10)' +' (AUC = %0.3f)' % roc_auc_CNB_ti_610)

x = np.linspace(0, 1, 5)
plt.plot(x, x, 'k--')

plt.legend(loc = "lower right")
plt.savefig("MS_plot_3_CNB.jpg")

print("Naive Bayes script completed at " + str(datetime.datetime.now()))