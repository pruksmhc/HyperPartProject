# Hyperpartisan news detection- preprocessing script
# Author: Yash Deshpande
# Last modified: November 23rd, 2018

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
		tr_unreadable_data.append(str(tr_item['@id']))
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
                val_unreadable_data.append(str(val_item['@id']))    
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
                tr_unreadable_labels.append(str(tr_labels_item['@id']))    
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
                val_unreadable_labels.append(str(val_labels_item['@id']))   
                continue

print ("Finished parsing validation labels. Completed at " + str(datetime.datetime.now()))

print ("Converting dictionaries to dataframes. Started at " + str(datetime.datetime.now()))

training_data_df = pd.DataFrame.from_dict(tr_articles_dict, orient = 'index')
validation_data_df = pd.DataFrame.from_dict(val_articles_dict, orient = 'index')
training_labels_df = pd.DataFrame.from_dict(tr_labels_dict, orient = 'index')
validation_labels_df = pd.DataFrame.from_dict(val_labels_dict, orient = 'index')


print ("Finished writing to dataframes at " + str(datetime.datetime.now()))

print ("Pickle-ing dataframes to memory. Started at " + str(datetime.datetime.now()))

n_bytes = 2**31
max_bytes = 2**31 - 1

tr_data = bytearray(n_bytes)
val_data = bytearray(n_bytes)
tr_labels = bytearray(n_bytes)
val_labels = bytearray(n_bytes)

file_training_df = open('training_data_df.obj', 'wb')
tr_bytes_out = pickle.dumps(tr_data)
with file_training_df as f_out:
    for idx in range(0, len(tr_bytes_out), max_bytes):
        f_out.write(tr_bytes_out[idx:idx+max_bytes])

file_validation_df = open('validation_data_df.obj', 'wb')
val_bytes_out = pickle.dumps(val_data)
with file_validation_df as f_out:
    for idx in range(0, len(val_bytes_out), max_bytes):
        f_out.write(val_bytes_out[idx:idx+max_bytes])

file_training_labels_df = open('training_labels_df.obj', 'wb')
trl_bytes_out = pickle.dumps(tr_labels)
with file_training_labels_df as f_out:
    for idx in range(0, len(trl_bytes_out), max_bytes):
        f_out.write(trl_bytes_out[idx:idx+max_bytes])

file_validation_labels_df = open('validation_labels_df.obj', 'wb')
val_l_bytes_out = pickle.dumps(val_labels)
with file_validation_labels_df as f_out:
    for idx in range(0, len(val_l_bytes_out), max_bytes):
        f_out.write(val_l_bytes_out[idx:idx+max_bytes])

file_unreadable_training_data = open('training_data_unreadable.obj', 'wb')
pickle.dump(tr_unreadable_data, file_unreadable_training_data)

file_unreadable_val_data = open('validation_data_unreadable.obj', 'wb')
pickle.dump(val_unreadable_data, file_unreadable_val_data)

file_unreadable_training_labels = open('training_labels_unreadable.obj', 'wb')
pickle.dump(tr_unreadable_labels, file_unreadable_training_labels)

file_unreadable_val_labels = open('val_labels_unreadable.obj', 'wb')
pickle.dump(val_unreadable_labels, file_unreadable_val_labels)

print ("Finished pickle-ing dataframes to memory at " + str(datetime.datetime.now()))

print ("Script completed at " + str(datetime.datetime.now()))                                                                                          