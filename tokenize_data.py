import nltk
import re
import os
import json
import zipfile
import random
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import TensorBoard
from transformers import BertTokenizerFast, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import multiprocessing


# Load the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Load the pre-trained BERT model
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)


os.makedirs('data/wordbroken', exist_ok=True)
os.makedirs('data/sentencebroken', exist_ok=True)
clean_df = pd.read_csv('data/clean/clean_stock_tweet_sentiment.csv')
clean_df['tweet'] = clean_df['tweet'].astype('str')
clean_df['tweet_cleaned'] = clean_df['tweet_cleaned'].astype('str')

print('------- tokenizing process ----------------')

# Define the function for parallel processing
def process_batch(batch):
    input_ids = tokenizer.batch_encode_plus(batch, add_special_tokens=False, return_token_type_ids=False).input_ids
    concatenated_input_ids = sum(input_ids, [])
    with open('data/wordbroken/data_raw.txt', 'a+') as f:
        f.write(' '.join(tokenizer.convert_ids_to_tokens(concatenated_input_ids)))
    return True

# Example input sentences
sentences = clean_df['tweet'].tolist()

# Split the input sentences into batches
batch_size = 1024
batches = [sentences[i:i+batch_size] for i in range(0, len(sentences), batch_size)]

# Create a multiprocessing pool
pool = multiprocessing.Pool()

# Process the batches in parallel
results = pool.map(process_batch, batches)

# Close the pool to release resources
pool.close()
pool.join()

# Define the function for parallel processing
def process_batch(batch):
    input_ids = tokenizer.batch_encode_plus(batch, add_special_tokens=False, return_token_type_ids=False).input_ids
    concatenated_input_ids = sum(input_ids, [])
    with open('data/wordbroken/data_cleaned.txt', 'a+') as f:
        f.write(' '.join(tokenizer.convert_ids_to_tokens(concatenated_input_ids)))
    return True

# Example input sentences
sentences = clean_df['tweet_cleaned'].tolist()

# Split the input sentences into batches
batch_size = 1024
batches = [sentences[i:i+batch_size] for i in range(0, len(sentences), batch_size)]

# Create a multiprocessing pool
pool = multiprocessing.Pool()

# Process the batches in parallel
results = pool.map(process_batch, batches)

# Close the pool to release resources
pool.close()
pool.join()

# Define the function for parallel processing
def process_batch(batch):
    input_ids = tokenizer.batch_encode_plus(batch, add_special_tokens=False, return_token_type_ids=False).input_ids
    concatenated_input_ids = sum(input_ids, [])
    with open('data/wordbroken/neg_data_cleaned.txt', 'a+') as f:
        f.write(' '.join(tokenizer.convert_ids_to_tokens(concatenated_input_ids)))
    return True

# Example input sentences
sentences = clean_df.query('sentiment == -1')['tweet_cleaned'].tolist()

# Split the input sentences into batches
batch_size = 1024
batches = [sentences[i:i+batch_size] for i in range(0, len(sentences), batch_size)]

# Create a multiprocessing pool
pool = multiprocessing.Pool()

# Process the batches in parallel
results = pool.map(process_batch, batches)

# Close the pool to release resources
pool.close()
pool.join()

# Define the function for parallel processing
def process_batch(batch):
    input_ids = tokenizer.batch_encode_plus(batch, add_special_tokens=False, return_token_type_ids=False).input_ids
    concatenated_input_ids = sum(input_ids, [])
    with open('data/wordbroken/pos_data_cleaned.txt', 'a+') as f:
        f.write(' '.join(tokenizer.convert_ids_to_tokens(concatenated_input_ids)))
    return True

# Example input sentences
sentences = clean_df.query('sentiment == 1')['tweet_cleaned'].tolist()

# Split the input sentences into batches
batch_size = 1024
batches = [sentences[i:i+batch_size] for i in range(0, len(sentences), batch_size)]

# Create a multiprocessing pool
pool = multiprocessing.Pool()

# Process the batches in parallel
results = pool.map(process_batch, batches)

# Close the pool to release resources
pool.close()
pool.join()

# Define the function for parallel processing
def process_batch(batch):
    input_ids = tokenizer.batch_encode_plus(batch, add_special_tokens=False, return_token_type_ids=False).input_ids
    concatenated_input_ids = sum(input_ids, [])
    with open('data/wordbroken/neu_data_cleaned.txt', 'a+') as f:
        f.write(' '.join(tokenizer.convert_ids_to_tokens(concatenated_input_ids)))
    return True

# Example input sentences
sentences = clean_df.query('sentiment == 0')['tweet_cleaned'].tolist()

# Split the input sentences into batches
batch_size = 1024
batches = [sentences[i:i+batch_size] for i in range(0, len(sentences), batch_size)]

# Create a multiprocessing pool
pool = multiprocessing.Pool()

# Process the batches in parallel
results = pool.map(process_batch, batches)

# Close the pool to release resources
pool.close()
pool.join()

def tokenize_tweet(tweet):
    return nltk.sent_tokenize(tweet)


sent_tokenized = clean_df['tweet'].apply(tokenize_tweet)

batch_size = 800

for i in range(0, sent_tokenized.size, batch_size):
  batch = sent_tokenized[i:i+batch_size]
  result = '\n'.join(sum(batch, []))
  with open('data/sentencebroken/data_raw.txt', 'a+') as f:
    f.write(result + '\n')

sent_tokenized = clean_df['tweet_cleaned'].apply(tokenize_tweet)

batch_size = 800

for i in range(0, sent_tokenized.size, batch_size):
  batch = sent_tokenized[i:i+batch_size]
  result = '\n'.join(sum(batch, []))
  with open('data/sentencebroken/data_cleaned.txt', 'a+') as f:
    f.write(result + '\n')
    
    
pos_sent_tokenized = clean_df.query('sentiment == 1')['tweet_cleaned'].apply(tokenize_tweet)
neg_sent_tokenized = clean_df.query('sentiment == -1')['tweet_cleaned'].apply(tokenize_tweet)
neu_sent_tokenized = clean_df.query('sentiment == 0')['tweet_cleaned'].apply(tokenize_tweet)

batch_size = 800

for i in range(0, pos_sent_tokenized.size, batch_size):
  batch = pos_sent_tokenized[i:i+batch_size]
  result = '\n'.join(sum(batch, []))
  with open('data/sentencebroken/pos_data_cleaned.txt', 'a+') as f:
    f.write(result + '\n')

for i in range(0, neg_sent_tokenized.size, batch_size):
  batch = neg_sent_tokenized[i:i+batch_size]
  result = '\n'.join(sum(batch, []))
  with open('data/sentencebroken/neg_data_cleaned.txt', 'a+') as f:
    f.write(result + '\n')

for i in range(0, neu_sent_tokenized.size, batch_size):
  batch = neu_sent_tokenized[i:i+batch_size]
  result = '\n'.join(sum(batch, []))
  with open('data/sentencebroken/neu_data_cleaned.txt', 'a+') as f:
    f.write(result + '\n')
    