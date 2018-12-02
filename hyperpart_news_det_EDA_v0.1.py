# Hyperpartisan news detection- exploratory analyses and basic modeling v0.1
# Author: Yash Deshpande
# Last modified: November 26th, 2018
# Updates:

import datetime

print("EDA script started at " + str(datetime.datetime.now()))

import pandas as pd 
import string
import numpy as np
from sklearn import model_selection as ms
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics

# Loading data and mapping character classes to numerics 

training_data_df = pd.read_csv('sample_data_labelled.csv.gz', sep = '|', header = 0)

training_data_df['Hyperpartisan'][training_data_df['Hyperpartisan'] == True] = 1
training_data_df['Hyperpartisan'][training_data_df['Hyperpartisan'] == False] = 0

X = training_data_df['Article']
Y = training_data_df['Hyperpartisan']
Y = Y.astype(int)
X_train, X_test, Y_train, Y_test = ms.train_test_split(X, Y, test_size = 0.25)

# Vectorizing inputs using TF-IDF
ti_vectorize_data = TfidfVectorizer(ngram_range=(5, 8), stop_words = 'english').fit(X)

x_train_vec_tfidf, x_test_vec_tfidf = ti_vectorize_data.transform(X_train), ti_vectorize_data.transform(X_test)

# Multinomial Naive Bayes model 
MBB_model_ti =  BernoulliNB()
MBB_model_ti.fit(x_train_vec_tfidf, Y_train)
preds_MBB_tfidf = MBB_model_ti.predict_proba(x_test_vec_tfidf)

fpr_MBB_ti, tpr_MBB_ti, thresholds_MBB_ti = metrics.roc_curve(Y_test, preds_MBB_tfidf[:,1])

roc_auc_MBB_ti = metrics.auc(fpr_MBB_ti, tpr_MBB_ti)

plt.figure(figsize = (20, 10))
plt.title("ROC curve- Multinomial Naive Bayes [n-gram range = (5, 8)]", fontsize = 20)
plt.ylabel("True positive rate (TPR)", size = 15)
plt.xlabel("False positive rate (FPR)", size = 15)

plt.plot(fpr_MBB_ti, tpr_MBB_ti, 'y', label = 'Bernoulli Naive Bayes (TF-IDF)' +' (AUC = %0.3f)' % roc_auc_MBB_ti)

x = np.linspace(0, 1, 5)
plt.plot(x, x, 'k--')

plt.legend(loc = "lower right")
plt.savefig("EDA_multinomial_bayes_plot_5.jpg")
print("EDA script completed at " + str(datetime.datetime.now()))