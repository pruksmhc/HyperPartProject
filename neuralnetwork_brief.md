After using basic models such as Logistic Regression and SVM(support vector machine) for text classification. Neural Network is used to see if we can get a better performance.  Word vectors are used here instead of word frequency. Word embeddings are vectorized representations of words. In order for the representations to be useful, some empirical conditions should be satisfied. For example, similar words should be closer to each other in the vector space. During the training process, the repersentations are constantly changing as well to optimize its performance.

 Highlighted steps in this Neural Network based approach: 

- Three columns in the dataframe are mainly used: ID, Article and Hyperpartisan. The whole dataset is splited into training and validation and testing using 0.75/0.125/0.125 ratio.
- The mean value for article length is calculated for the padding purpose.
- A function called **preproc** is used to tokenize articles, lemmatization is used, stop words and puctuations are removed before tokenizing articles using n grams. After this step, each article becomes a list of n-gram tokens and all the n-gram tokens in the training dataset are recorded to build the dictionary afterwards.
- Uni-gram is used in our model and the maximum vocabulary size is set to be 20000. According to the gram frequency, the first 20000 grams are selected. **id2token** is a list of tokens, where id2token[i] returns token that corresponds to token i. **token2id** is a dictionary where keys represent tokens and corresponding values represent indices. 0,1 are used as padding index and unknown index respectively. 
- Converting token to id in the dataset using the built dictionary. Since neural network prefers well defined fixed-length inputs and outputs, articles are padded and the whole training dataset is transformed into minibatches so that the input articles can be fed in parallel to the network to speed up the computing.
- In the neural network architecture, after the embedding layer, a linear layer is used to convert the output to dimension 2. After softmax, the two numbers corresponding to the probabilities of belonging to each class.
- CrossEntropyLoss is used, and Adam is used as the optimizer. Other hyper-parameters like learining rate, embedding size are tuned afterwards using the validation dataset. 
- ablation study: 

- A function called **preproc** is used to tokenize articles, lemmatization is used, stop words and puctuations are removed before tokenizing articles using n grams with n = 1 (in our model). After this step, each article becomes a list of n-gram tokens and all the n-gram tokens in the training dataset are recorded to build the dictionary afterwards.

- Uni-gram is used in our model and the maximum vocabulary size is set to be 20000. By vocaulbary size, we select the first  According to the frequency, the first 20000 most frequent unigrams are selected.  **id2token** is a list of tokens, where id2token[i] returns token that corresponds to token i. **token2id** is a dictionary where keys represent tokens and corresponding values represent indices. 0,1 are used as padding index and unknown index respectively.

- Converting token to id in the dataset using the tokenize2dataset function, which are used to index the words, which the pytorch neural network expects (which it then uses to index the embedding matrix which is the first layer of the neural network).Since neural network necessitates fixed-length inputs and outputs, articles are padded and the whole training dataset is transformed into minibatches so that the input articles can be fed in parallel to the network to speed up the computing. We use pad_packed_function to speed up processing, which basically ignores and accounts for the difference in lengths of various inputs in the batch and thus reduces computation time. 

- In the neural network architecture, after the embedding layer (which transforms each index into a d-dimensional vector), the word vectors for the input sequence are summed up to get a 1xd dimensional vector.A a linear layer is used to convert the output to dimension 2. After softmax, the two numbers corresponding to the probabilities of belonging to each class (hyperpartisan or not). 

- CrossEntropyLoss is used, and Adam is used as the optimizer due to research papers which show that Adam has been shown to converge. Other hyper-parameters like hidden size are tuned afterwards using the validation dataset as a marker. 

As the first approach, the bag of words approach, after obtaining the vector representation of each word in the sentence via an embedding layer, those vectors are summed up element-wise to represent the encoded information of the input. The best performance using this method is 82% on validation and 73.2% on test set. Further to this, another approach is also applied. Because of the length of the article, it may not be a good idea to add all the information in the article as one vector, summing up the embeddings element-wise. Instead of the bag-of-words model, we used a Recurrent Neural Network. A Recurrent Neural Network is a network which relies on not only the input, but also the previous hidden state. It thus carries a memory portion and can be fed in input sequentially. Thus, it takes into account the order of words, which is important in such a nuanced task such as detecting hyperpartisanship. We carried on an ablation studying on embeding size and learning rates on our BOW approach. Due to the time limit, we use the best parameter combination we obtained tuning the previous neural network in bidirectional GRU, using learning rate 0.01, embedding size 300, number of layer 3, followed by two fully connected layer. The accuracy is **87.8%**. We did not have time to do an ablation study for the recurrent neural network model. 


Ablation study:
	We tuned for several hyperparameters - word embedding size and learning rate. Our default parameters were 0.001 learning rate and 300 embedding size. 
	For our embedding size, we decided to try an embedding size of 200 and 300, and for learning rate 0.01 and 3e-4, which has been shown to perform well for Adam optimization. 

Test scores (Scores of testing on the test set)
Embedding size   
200 - 60.3
300 - 70.5

Learning rate: 
0.001 - 73.2
0.0003 - 64.3

We were able to achieve a maximum of 73.2% accuracy (NLLLoss) using neural network. 
We can see that the test scores were worse for neural network after testing on the test data, which is made up of articles from different publications, although not as drastic of a decrease in comparison to Naive Bayes and SVM. In terms of parameters, embedding size significantly increased performance. This is because in our BOW model, the embedding vectors of each input is summed together before being fed into a Linear layer. Thus, the summation of the vectors must encode the semantics of the article. A higher dimensin means that more information will be preserved, and thus the RNN is able to tweak the NN weights to make a decision more efficiently. 

Meanwhile, higher learning rate led to higher performance. However, as seen in the graphs above, training with higher learning rate led to higher instability during training, as the optimizer first overshot the minimum in the loss function manifold (as shown by the drastic increase and decrease in validation set preformance in the first epoch).

Further work: 
-Train on more embedding sizes
-Try linear annealing for learning rate 
-Try more than 1 n-gram. 


