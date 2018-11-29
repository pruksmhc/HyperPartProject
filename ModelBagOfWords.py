# First import torch related libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import pdb
import numpy as np 
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from HyperPartDataLoader import *

MAX_SENTENCE_LENGTH = 19650
class BagOfWords(nn.Module):
    """
    BagOfWords classification model
    """
    def __init__(self, vocab_size, emb_dim):
        """
        @param vocab_size: size of the vocabulary. 
        @param emb_dim: size of the word embedding
        # Binary calssification
        """
        super(BagOfWords, self).__init__()
        # pay attention to padding_idx 
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.linear = nn.Linear(emb_dim,2)
    
    def forward(self, data, length):
        """
        
        @param data: matrix of size (batch_size, max_sentence_length). Each row in data represents a 
            review that is represented using n-gram index. Note that they are padded to have same length.
        @param length: an int tensor of size (batch_size), which represents the non-trivial (excludes padding)
            length of each sentences in the data.
        """
        out = self.embed(data)
        out = torch.sum(out, dim=1)
        # view basically reshapes it, so this averages it out. 
        out /= length.view(length.size()[0],1).expand_as(out).float()
     
        # return logits
        out = self.linear(out.float())
        return out

emb_dim = 100
#model = BagOfWords(len(id2token), emb_dim)

learning_rate = 0.01
num_epochs = 3 # number epoch to train


# Function for testing the model
def test_model(loader, model):
    """
    Help function that tests the model's performance on a dataset
    @param: loader - data loader for the dataset to test against
    """
    correct = 0
    total = 0
    model.eval()
    for data, lengths, labels in loader:
        data_batch, length_batch, label_batch = data, lengths, labels
        outputs = F.softmax(model(data_batch, length_batch), dim=1)
        predicted = outputs.max(1, keepdim=True)[1]
        total += labels.size(0)
        correct += predicted.eq(labels.view_as(predicted)).sum().item()
    return (100 * correct / total)

def test_model_routine(train_loader, val_loader, model, criterion, optimizer, num_epochs, learning_rate, scheduler=None):
    acc_per_step_val = []
    acc_per_step_train = []
    for epoch in range(num_epochs):
        acc_per_epoch = []
        acc_per_epoch_val = []
        acc = []
        for i, (data, lengths, labels) in enumerate(train_loader):
            print(i)
            model.train()
            data_batch, length_batch, label_batch = data, lengths, labels
            optimizer.zero_grad()
            outputs = model(data_batch, length_batch)
            loss = criterion(outputs, label_batch)
            loss.backward(retain_graph=True)
            optimizer.step()
            # validate every 100 iterations
            if i > 0 and i % 100 == 0:
                # validate
                val_acc = test_model(val_loader, model) 
                train_acc = test_model(train_loader, model)
                acc.append(val_acc)
                acc_per_epoch_val.append(val_acc)
                acc_per_epoch.append(train_acc)
                print('Epoch: [{}/{}], Step: [{}/{}], Validation Acc: {}, Train Acc: {}'.format( 
                           epoch+1, num_epochs, i+1, len(train_loader), val_acc, train_acc))
        #scheduler.step(loss)
        print("Average accuracy is"+ str(np.mean(acc)))
        acc_per_step_val.append(acc_per_epoch_val)
        acc_per_step_train.append(acc_per_epoch)
    print("total average accuarcies validation")
    print(acc_per_step_val)
    print("total accuracies train")
    print(acc_per_step_train)
    save_model(model, acc_per_step_val, acc_per_step_train, "Bag-of-words Deep Learning Model Performance on HyperPartisan Task")
    return acc_per_step_val, acc_per_step_train, model

def save_model(model, val_accs, train_accs, title):
    pdb.set_trace()
    val_accs = np.array(val_accs)
    max_val = val_accs.max()
    train_accs = np.array(train_accs)
    link =  ""
    torch.save(model.state_dict(), link + "model_states")
    pickle.dump(val_accs, open(link + "val_accuracies", "wb"))
    pickle.dump(train_accs, open(link + "train_accuracies", "wb"))
    pickle.dump(max_val, open(link + "maxvalaccis"+str(max_val), "wb"))
    # this is when you want to overlay
    num_in_epoch = np.shape(train_accs)[1]
    num_epochs = np.shape(train_accs)[0]
    x_vals = np.arange(0, num_epochs, 1.0/float(num_in_epoch))
    fig = plt.figure()
    plt.title(title)
    plt.plot(x_vals, train_accs.flatten(), label="Training Accuracy")
    plt.plot(x_vals, val_accs.flatten(), label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.ylabel("Accuracy of Model With Given Parameter")
    plt.xlabel("Epochs (Batch Size 32)")
    plt.ylim(0,100)
    plt.xlim(0, num_epochs)
    plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    plt.xticks(np.arange(num_epochs + 1))
    fig.savefig(link+"graph.png")

learning_rate = 0.001
num_epochs = 10 # number epoch to train
BATCH_SIZE = 32
max_vocab_size = 20002
# Criterion and Optimizer
model = BagOfWords(max_vocab_size, 100)
# Criterion and Optimizer
criterion = torch.nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_data_indices = pickle.load(open('data/train_data_indexed', "rb"))
val_data_indices = pickle.load(open('data/val_data_indexed', "rb"))

train_labels = pd.read_pickle('data/train_labels').tolist()
val_labels = pd.read_pickle('data/val_labels').tolist()

train_data_indices = train_data_indices[:200]
val_data_indices  = val_data_indices[:100]
train_labels = train_labels[:200]
val_labels = val_labels[:100]

convert_to_binary = {True: 1, False: 0}
train_labeldf = [convert_to_binary[x] for x in train_labels]
val_labeldf = [convert_to_binary[x] for x in val_labels]

train_dataset = HyperPartGroupDataset(train_data_indices, train_labeldf)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=BATCH_SIZE,
                                           collate_fn=hype_collate_func,
                                           shuffle=True)

val_dataset = HyperPartGroupDataset(val_data_indices, val_labeldf)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                           batch_size=BATCH_SIZE,
                                           collate_fn=hype_collate_func,
                                           shuffle=True)


test_model_routine(train_loader, val_loader, model, criterion, optimizer, num_epochs, learning_rate)       

