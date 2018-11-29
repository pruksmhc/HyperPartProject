# Hyperpartisan news detection- text pre-processing script v0.1
# Author: Yash Deshpande
# Last modified: November 26th, 2018
# Updates:
# 1. Merged with labels (note: nans present, TBC.)

import datetime

print("Pre-processing script started at " + str(datetime.datetime.now()))

import pandas as pd
import string
import re

print("Reading in raw data. Started at " + str(datetime.datetime.now()))

training_data_df_1 = pd.read_csv('training_data_pck_01.csv.gz', sep = '|', header = 0)
training_data_df_1.columns = ['ID', 'Article']
training_data_df_2 = pd.read_csv('training_data_pck_02.csv.gz', sep = '|', header = 0)
training_data_df_2.columns = ['ID', 'Article']

full_training_data_df = pd.concat([training_data_df_1, training_data_df_2])
full_training_data_df.index = list(range(len(full_training_data_df)))

print("Finished reading in raw data. Completed at " + str(datetime.datetime.now()))

print("Cleaning data. Started at " + str(datetime.datetime.now()))

table = str.maketrans({key: None for key in string.punctuation})
stopwords = ['OrderedDicta', 'OrderedDicthref', 'href', 'type', 'text']

for i in range(len(full_training_data_df)):
    print("Processing row %d " % (i + 1))
    full_training_data_df['Article'][i] = [str(str(full_training_data_df['Article'][i]).split()).translate(table).replace(stopword, '') for stopword in stopwords]

training_labels_df = pd.read_csv('training_labels_pck.csv.gz', sep = '|', header = 0)
training_labels_df.columns = ['ID', 'Hyperpartisan']

print("Cleaning data. Started at " + str(datetime.datetime.now()))

print("Merging with labels and unloading clean data. Started at " +str(datetime.datetime.now()))

full_training_data_labels_df = full_training_data_df.merge(training_labels_df, how = 'inner', on = 'ID')

len_split = int(len(full_training_data_df)/2)

full_training_data_labels_1 = full_training_data_labels_df.head(len_split)
full_training_data_labels_2 = full_training_data_labels_df.tail(len_split)

full_training_data_labels_1.to_csv("processed_training_data_01.csv", sep = '|', chunksize = 10000)
full_training_data_labels_2.to_csv("processed_training_data_02.csv", sep = '|', chunksize = 10000)

print("Finished merging with labels and unloading clean data. Completed at " +str(datetime.datetime.now()))

print("Pre-processing script completed at " + str(datetime.datetime.now()))