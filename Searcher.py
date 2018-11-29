import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np 
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from RNN_Class import *
from CNNModel import *


class HyperSearch(nn.Module):
	"""
		the purpose of this is to save the models with the accuracy of the validation (the maximum), 
		and the hyperparameter seetings, to get the model with the best performance. 
	"""
	def __init__(self, dirlink, num_epochs, lr):
		self.dirlink = dirlink # where to save
		self.num_epochs = num_epochs
		self.lr = lr

	def test_model(self, loader, model):
	    """
	    Help function that tests the model's performance on a dataset
	    @param: loader - data loader for the dataset to test against
	    """
	    correct = 0
	    total = 0
	    model.eval()
	    for sentence1, sentence2, length1, length2, order_1, order_2, labels in loader:
	        outputs = model(sentence1, sentence2, length1, length2, order_1, order_2)
	        outputs = F.softmax(outputs, dim=1)
	        predicted = outputs.max(1, keepdim=True)[1]
	        total += labels.size(0)
	        correct += predicted.eq(labels.view_as(predicted)).sum().item()
	    return (100 * correct / total)

	def save_model(self, model, val_accs, train_accs, RNN_or_CNN, param_name, param_val, title):
		pdb.set_trace()
		val_accs = np.array(val_accs)
		max_val = val_accs.max()
		train_accs = np.array(train_accs)
		link = self.dirlink + "/"+RNN_or_CNN +"/"+ param_name+"/" + str(param_val)+"/"
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
		# What if I actually do the graphing here, then it would output the graphs here. 


	def search_parameters(self,param_name, CNN_or_RNN, param_values, other_values, train_loader, val_loader, template_title):
		"""
		other_values should be the default values, with the parameters that are changing
		CNN_or_RNN to import various RNN and CNNs
		Here, we do the train_loader nd val_loader from the original script
		So the preprocesisng should alreadyd be done at this point. 

		"""
		criterion = torch.nn.CrossEntropyLoss()
		total_step = len(train_loader)
		for param_val in param_values:
			val_accs = []
			train_accs = []
			other_values[param_name] = param_val
			if CNN_or_RNN == "CNN":
				#model = CNN(emb_size=300, hidden_size=600,  num_classes=3, vocab_size=len(current_word2idx), kernel_size =3, weight=torch.FloatTensor(weights), 50, 28)
				model = CNN(**other_values)
			elif CNN_or_RNN == "RNN":
				#model = RNN(emb_size=300, hidden_size=600, num_layers=1, num_classes=3,  weight=torch.FloatTensor(weights))
				model = RNN(**other_values)
			else:
				model = CNNWithDropout(**other_values)
			optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
			for epoch in range(self.num_epochs):
				v_accs = []
				t_accs = []
				for i, (sentence1, sentence2, length1, length2, order_1, order_2, labels) in enumerate(train_loader):
					model.train()
					optimizer.zero_grad()
					outputs = model(sentence1, sentence2, length1, length2, order_1, order_2)
					loss = criterion(outputs, labels)
					loss.backward()
					optimizer.step()
					parameters = list(filter(lambda p: p.grad is not None, model.parameters()))
					norm_type = 2
					total_norm = 0
					# we should have 10 in between. so 10 per epoch. 
					if i > 0 and i % 100 == 0:
						# there should be 2 per epoch
						print("Parameter "+param_name)
						print(param_val)
						s_outputs = F.softmax(outputs, dim=1)
						total = labels.size(0)
						predicted = s_outputs.max(1, keepdim=True)[1]
						correct = predicted.eq(labels.view_as(predicted)).sum().item()
						train_acc = (100 * correct / total)
						print("Train accuracy is" + str(train_acc))
						t_accs.append(train_acc)
						val_acc = self.test_model(val_loader, model)
						print('Epoch: [{}/{}], Step: [{}/{}], Validation Acc: {}'.format(
			                       epoch+1, self.num_epochs, i+1, total_step, val_acc))
						v_accs.append(val_acc)
				val_accs.append(v_accs)
				train_accs.append(t_accs)
			title = template_title + str(param_val)
			self.save_model(model, val_accs, train_accs, CNN_or_RNN, param_name, param_val, title)

