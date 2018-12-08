# Hyperpartisan News Detection

This project aims to detect partisanship in written news media, tackling it as a binary text classification problem.

It is a collaborative effort undertaken both as part of a course project at the NYU Center for Data Science, and SemEval 2019 Task 4. Group members include:

* Yash Deshpande (yd1282@nyu.edu)
* Yada Pruksachatkun (yp913@nyu.edu)
* Aja Klevs (ak7288@nyu.edu)
* Jiayi Du (jd4138@nyu.edu)

This project is a work in progress. As of this update, we have reached a classification accuracy of **80.2%**, using a recurrent neural network architecture with a learning rate of **0.01** and an embedding size of **450**. The data used to train is a collection of **20,000** articles labelled as **1**/**0** (partisan/non-partisan) based on the general tendency of the publisher. Baseline models such as logistic regression, support vector machines, and Naive Bayes approaches failed to achieve a high out-of-sample accuracy, mose likely because the test set and training set did not have any publishers in common, i.e. the distributions of data in the training and test sets were very different.

Please see the updated project report, titled **"Project Report- Hyperpartisan News Detection v1.0.pdf"** for a detailed description of our modeling process.
