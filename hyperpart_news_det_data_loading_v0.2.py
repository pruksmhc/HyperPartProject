# Hyperpartisan news detection- data loading script v0.2
# Author: Yash Deshpande
# Last modified: November 23rd, 2018
# Updates:
# 1. pandas' to_pickle used instead of pickle.dump

import datetime
print ("Script started at " + str(datetime.datetime.now()))

import pickle
import xmltodict as xtd
from collections import defaultdict
import pandas as pd
import gzip 

print ("Parsing training dataset. Started at " + str(datetime.datetime.now()))
tr_articles_dict = defaultdict(list)
tr_unreadable_data = []

with gzip.open('articles-training-bypublisher-20181122.xml.gz') as r1:
	dict_training_data = xtd.parse(r1.read(), xml_attribs = True)


for tr_item in dict_training_data['articles']['article']:
	try:
		tr_articles_dict[tr_item['@id']].append(tr_item['p'])
	except:
		tr_unreadable_data.append(str(tr_item))
		continue	

print ("Finished parsing training dataset. Completed at " + str(datetime.datetime.now()))

print ("Parsing validation dataset. Started at " + str(datetime.datetime.now()))
val_articles_dict= defaultdict(list)
val_unreadable_data = []

with gzip.open('articles-validation-bypublisher-20181122.xml.gz') as r2:
        dict_validation_data = xtd.parse(r2.read(), xml_attribs = True)

for val_item in dict_validation_data['articles']['article']:
        try:
                val_articles_dict[val_item['@id']].append(val_item['p'])
        except:
                val_unreadable_data.append(str(val_item))
                continue

print ("Finished parsing validation dataset. Completed at " + str(datetime.datetime.now()))

print ("Parsing training labels. Started at " + str(datetime.datetime.now()))
tr_labels_dict = defaultdict(list)
tr_unreadable_labels = []

with gzip.open('ground-truth-training-20180831.xml.gz') as r3:
        dict_training_labels = xtd.parse(r3.read(), xml_attribs = True)


for tr_labels_item in dict_training_labels['articles']['article']:
        try:
                tr_labels_dict[tr_labels_item['@id']].append(tr_labels_item['@hyperpartisan'])
        except:
                tr_unreadable_labels.append(str(tr_labels_item))
                continue

print ("Finished parsing training labels. Completed at " + str(datetime.datetime.now()))
 
print ("Parsing validation labels. Started at " + str(datetime.datetime.now())) 
val_labels_dict = defaultdict(list)
val_unreadable_labels = []  

with gzip.open('articles-validation-bypublisher-20181122.xml.gz') as r4:
        dict_validation_labels = xtd.parse(r4.read(), xml_attribs = True)

for val_labels_item in dict_validation_labels['articles']['article']:
        try:
                val_labels_dict[val_labels_item['@id']].append(val_labels_item['@hyperpartisan'])
        except:
                val_unreadable_labels.append(str(val_labels_item))
                continue

print ("Finished parsing validation labels. Completed at " + str(datetime.datetime.now()))

print ("Converting dictionaries to dataframes. Started at " + str(datetime.datetime.now()))

training_data_df = pd.DataFrame.from_dict(tr_articles_dict, orient = 'index')
validation_data_df = pd.DataFrame.from_dict(val_articles_dict, orient = 'index')
training_labels_df = pd.DataFrame.from_dict(tr_labels_dict, orient = 'index')
validation_labels_df = pd.DataFrame.from_dict(val_labels_dict, orient = 'index')

print ("Finished writing to dataframes at " + str(datetime.datetime.now()))

print ("Pickle-ing dataframes to memory. Started at " + str(datetime.datetime.now()))

training_data_df.to_pickle('training_data_pck.pickle')
validation_data_df.to_pickle('val_data_pck.pickle')
training_labels_df.to_pickle('training_labels_pck.pickle')
validation_labels_df.to_pickle('val_labels_pck.pickle')

pd.DataFrame(tr_unreadable_data).to_pickle('err_training_data_pck.pickle')
pd.DataFrame(val_unreadable_data).to_pickle('err_val_labels_data.pickle')
pd.DataFrame(tr_unreadable_labels).to_pickle('err_training_labels_pck.pickle')
pd.DataFrame(val_unreadable_labels).to_pickle('err_val_labels_pck.pickle')

print ("Finished pickle-ing dataframes to memory at " + str(datetime.datetime.now()))

print ("Script completed at " + str(datetime.datetime.now()))                                                                                          