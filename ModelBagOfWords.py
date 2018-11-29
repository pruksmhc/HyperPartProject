# First import torch related libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

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

# Criterion and Optimizer
criterion = torch.nn.CrossEntropyLoss()  
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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
        acc = []
        for i, (data, lengths, labels) in enumerate(train_loader):
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
                acc_per_step_val.append(val_acc)
                acc_per_step_train.append(train_acc)
                print('Epoch: [{}/{}], Step: [{}/{}], Validation Acc: {}, Train Acc: {}'.format( 
                           epoch+1, num_epochs, i+1, len(train_loader), val_acc, train_acc))
        #scheduler.step(loss)
        print("Average accuracy is"+ str(np.mean(acc)))
    print("total average accuarcies validation")
    print(acc_per_step_val)
    print("total accuracies train")
    print(acc_per_step_train)
    return acc_per_step_val, acc_per_step_train, model
#test_model_routine(train_loader, val_loader, model, criterion, optimizer, num_epochs, learning_rate)       

