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


os.environ['KAGGLE_CONFIG_DIR'] = os.getcwd()
from kaggle.api.kaggle_api_extended import KaggleApi

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

stop_words = stopwords.words('english')

sia = SentimentIntensityAnalyzer()


# Set up the Kaggle API
api = KaggleApi()
api.authenticate()

dataset_name = "chenxidong/stock-tweet-sentiment-dataset"
api.dataset_download_files(dataset_name)

# Extract the ZIP file
with zipfile.ZipFile(f"{dataset_name.split('/')[1]}.zip", "r") as zip_ref:
    zip_ref.extractall("data/raw")


# Load the StockTwits sentiment dataset
df = pd.read_csv("data/raw/stock_tweet_sentiment.csv")
del df[df.columns[0]]
df = df.rename(columns={'timestamp': 'post_date'})
df = df.rename(columns={'text': 'tweet'})
df = df.rename(columns={'Sentiment': 'sentiment'})
df = df.rename(columns={'company_names': 'ticker_symbol'})
df = df[['post_date', 'tweet', 'sentiment', 'ticker_symbol']]


def get_sentiment(tweet):
    sentiment = sia.polarity_scores(tweet)
    score = sentiment['compound']
    if score > 0.5:
        return 1          #'positive'
    elif score < -0.5:
        return -1         #'negative'
    else:
        return 0          #'neutral'

def apply_get_sentiment(example):
    # get the tweet text from the example
    tweet_text = example['body']
    
    # apply the get_sentiment() function to the tweet text and store the result in a new 'sentiment' key
    example['sentiment'] = get_sentiment(tweet_text)

    return example


dataset = load_dataset("mjw/stock_market_tweets")
dataset = dataset['train'].shuffle().select(range(500000))
num_proc = multiprocessing.cpu_count()
dataset = dataset.map(apply_get_sentiment, num_proc=num_proc)

tmp_df = dataset.to_pandas()
tmp_df = tmp_df.rename(columns={'body': 'tweet'})
tmp_df = tmp_df[['post_date', 'tweet', 'ticker_symbol', 'sentiment']]
df = pd.concat([df, tmp_df], ignore_index=True)

dataset_name = "andreaskonradsen/stocks-with-sentiment-and-emotion-analysis"
api.dataset_download_files(dataset_name)

with zipfile.ZipFile(f"{dataset_name.split('/')[1]}.zip", "r") as zip_ref:
    zip_ref.extractall("data/raw")

def get_sentiment(probability):
    probability = probability.replace("'", "\"")
    probability = json.loads(probability)
    label = probability['label']
    if label == 'neutral':
      return 0
    elif label == 'positive':
      return 1
    elif label == 'negative':
      return -1
  
tmp_df = pd.read_csv("data/raw/tweetnlp_sentiment_emotion.csv")
tmp_df = tmp_df[:50000]
tmp_df = tmp_df.rename(columns={'body_cleaned': 'tweet'})
tmp_df['sentiment'] = tmp_df['probabilities'].apply(get_sentiment)
tmp_df = tmp_df[['post_date', 'tweet', 'sentiment']]
df = pd.concat([df, tmp_df], ignore_index=True)


dataset_name = "fabioturazzi/cryptocurrency-tweets-with-sentiment-analysis"
api.dataset_download_files(dataset_name)

with zipfile.ZipFile(f"{dataset_name.split('/')[1]}.zip", "r") as zip_ref:
    zip_ref.extractall("data/raw")

def get_sentiment(score):
    if score >= 0.5:
        return 1          #'positive'
    elif score <= -0.5:
        return -1         #'negative'
    else:
        return 0          #'neutral'
  
tmp_df = pd.read_csv("data/raw/tweets_sentiment.csv")
tmp_df = tmp_df[:50000]
tmp_df = tmp_df.rename(columns={'created_at': 'post_date'})
tmp_df['sentiment'] = tmp_df['compound'].apply(get_sentiment)
tmp_df = tmp_df[['post_date', 'tweet', 'sentiment']]
df = pd.concat([df, tmp_df], ignore_index=True)

os.makedirs('data/stats', exist_ok=True)

total_tweets = len(df)
negative_tweets = len(df.query("sentiment == -1"))
neutral_tweets = len(df.query("sentiment == 0"))
positive_tweets = len(df.query("sentiment == 1"))

print("totla tweets: ", len(df))
print("tweets with negative sentiment: ", len(df.query("sentiment == -1")))
print("tweets with neutral sentiment: ", len(df.query("sentiment == 0")))
print("tweets with positive sentiment: ", len(df.query("sentiment == 1")))

#save in csv
tmp_df = pd.DataFrame()
tmp_df['total_tweets'] = [total_tweets]
tmp_df['negative_tweets'] = [negative_tweets]
tmp_df['neutral_tweets'] = [neutral_tweets]
tmp_df['positive_tweets'] = [positive_tweets]
tmp_df.to_csv('data/stats/sentiment_distribution_befor_normalizing.csv')

plt.figure(figsize=(8, 6))
labels = ['Total tweets', 'Negative sentiment', 'Neutral sentiment', 'Positive sentiment']
values = [total_tweets, negative_tweets, neutral_tweets, positive_tweets]
plt.bar(labels, values, width=0.5)
plt.ylabel('Count')
plt.title('Sentiment Distribution')

plt.savefig('data/stats/sentiment_distribution_befor_normalizing.png')

negative_df = df.query("sentiment == -1").sample(n=40000)
neutral_df = df.query("sentiment == 0").sample(n=40000)
positive_df = df.query("sentiment == 1").sample(n=40000)
df = pd.concat([negative_df, neutral_df, positive_df], ignore_index=True)


total_tweets = len(df)
negative_tweets = len(df.query("sentiment == -1"))
neutral_tweets = len(df.query("sentiment == 0"))
positive_tweets = len(df.query("sentiment == 1"))

print("totla tweets: ", len(df))
print("tweets with negative sentiment: ", len(df.query("sentiment == -1")))
print("tweets with neutral sentiment: ", len(df.query("sentiment == 0")))
print("tweets with positive sentiment: ", len(df.query("sentiment == 1")))

#save in csv
tmp_df = pd.DataFrame()
tmp_df['total_tweets'] = [total_tweets]
tmp_df['negative_tweets'] = [negative_tweets]
tmp_df['neutral_tweets'] = [neutral_tweets]
tmp_df['positive_tweets'] = [positive_tweets]
tmp_df.to_csv('data/stats/sentiment_distribution.csv')


plt.figure(figsize=(8, 6))
labels = ['Total tweets', 'Negative sentiment', 'Neutral sentiment', 'Positive sentiment']
values = [total_tweets, negative_tweets, neutral_tweets, positive_tweets]
plt.bar(labels, values, width=0.5)
plt.ylabel('Count')
plt.title('Sentiment Distribution')
plt.savefig('data/stats/sentiment_distribution.png')

df.to_csv("data/raw/raw_stock_tweet_sentiment.csv")

def preprocess(text):
    # Remove non-Latin characters
    text = re.sub(r'[^\u0000-\u007F]+', '', text)
    # Remove mentions (@username)
    text = re.sub(r'@\w+', '', text)
    # Remove URLs
    text = re.sub(r'http.?://[^\s]+[\s]?', '', text)
    # Remove numbers and punctuation
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Convert to lowercase and tokenize
    text = text.lower().split()
    # Remove stop words
    text = [word for word in text if word not in stop_words]
    # Join the tokens back into a string
    text = ' '.join(text)
    return text


df['tweet'] = df['tweet'].astype('str')
df['tweet_cleaned'] = df['tweet'].apply(preprocess)

os.makedirs('data/clean/', exist_ok=True)
df.to_csv("data/clean/clean_stock_tweet_sentiment.csv")


# Load the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Load the pre-trained BERT model
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)


os.makedirs('data/wordbroken', exist_ok=True)
os.makedirs('data/sentencebroken', exist_ok=True)
clean_df = pd.read_csv('data/clean/clean_stock_tweet_sentiment.csv')
clean_df['tweet'] = clean_df['tweet'].astype('str')
clean_df['tweet_cleaned'] = clean_df['tweet_cleaned'].astype('str')

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
    
    
# Group by Company and Sentiment, and calculate the count
_df = clean_df.dropna(subset=['ticker_symbol'])

# Filter rows where the 'symbols' column is not empty
_df = clean_df.query("ticker_symbol != ''")

# show statistics about top-5 company
symbols = ["AAPL", "GOOG", "MSFT", "TSLA", 'AMZN']
_df = clean_df.query("ticker_symbol.isin(@symbols)")

grouped = _df.groupby(['ticker_symbol', 'sentiment']).size().unstack(fill_value=0)

# Plotting the bar chart
plt.rcParams['figure.figsize'] = [10, 6]
grouped.plot(kind='bar', stacked=True)

# Set chart title and labels
plt.title('Sentiment Analysis of Stock Tweets by Company')
plt.xlabel('Company')
plt.ylabel('Count')
plt.savefig('data/stats/top_5_companies_sentiment_distribution.png')


# Group by Company and Sentiment, and calculate the count
_df = clean_df.dropna(subset=['ticker_symbol'])

# Filter rows where the 'symbols' column is not empty
_df = clean_df.query("ticker_symbol != ''")

# show statistics about top-5 company
exclude_symbols  = ["AAPL", "GOOG", "MSFT", "TSLA", 'AMZN']
_df = clean_df.query("ticker_symbol.isin(@exclude_symbols)==False")

grouped = _df.groupby(['ticker_symbol', 'sentiment']).size().unstack(fill_value=0)

# Plotting the bar chart
plt.rcParams['figure.figsize'] = [100, 30]
grouped.plot(kind='bar', stacked=True)

# Set chart title and labels
plt.title('Sentiment Analysis of Stock Tweets by Company')
plt.xlabel('Company')
plt.ylabel('Count')
plt.savefig('data/stats/other_companies_sentiment_distribution.png')


cleaned_tokens = None
raw_tokens = None
with open('data/wordbroken/data_cleaned.txt', 'r') as f:
  cleaned_tokens = np.array(f.read().split(' '))

with open('data/wordbroken/data_raw.txt', 'r') as f:
  raw_tokens = np.array(f.read().split(' '))


raw_count = len(raw_tokens)
cleaned_count = len(cleaned_tokens)


print("number of tokens before pre_processing: ",raw_count)
print("number of tokens after pre_processing: ",cleaned_count)

labels = ['Before preprocessing', 'After preprocessing']
values = [raw_count, cleaned_count]

#save in csv file
tmp_df = pd.DataFrame()
tmp_df['before_preprocessing'] = [raw_count]
tmp_df['after_preprocessing'] = [cleaned_count]
tmp_df.to_csv('data/stats/token_counts_befor_after_pre_processing.csv')

# Create a new figure object
fig, ax = plt.subplots(figsize=(8, 6))

# Create a bar chart
ax.bar(labels, values, width=0.3)

# Set the y-axis to have 5 steps
ax.yaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

# Customize the chart
ax.set_ylabel('Token count')
ax.set_title('Token count before and after preprocessing')

# Save the chart to a PNG file
fig.savefig('data/stats/token_counts_befor_after_pre_processing.png', dpi=300)


cleaned_sentences_num = 0
raw_sentences_num = 0
with open('data/sentencebroken/data_cleaned.txt', 'r') as f:
  cleaned_sentences_num = sum(1 for _ in f)

with open('data/sentencebroken/data_raw.txt', 'r') as f:
  raw_sentences_num = sum(1 for _ in f)


print("number of sentences before pre_processing: ", raw_sentences_num)
print("number of sentences after pre_processing: ", cleaned_sentences_num)

labels = ['Before preprocessing', 'After preprocessing']
values = [raw_sentences_num, cleaned_sentences_num]

#save in csv file
tmp_df = pd.DataFrame()
tmp_df['before_preprocessing'] = [raw_sentences_num]
tmp_df['after_preprocessing'] = [cleaned_sentences_num]
tmp_df.to_csv('data/stats/sentence_counts_befor_after_pre_processing.csv')

fig, ax = plt.subplots(figsize=(8, 6))

# Create a bar chart
ax.bar(labels, values, width=0.3)

# Set the y-axis to have 5 steps
ax.yaxis.set_major_locator(MaxNLocator(6))
ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

# Customize the chart
ax.set_ylabel('Sentence count')
ax.set_title('Sentence count before and after preprocessing')

# Display the chart
fig.savefig('data/stats/sentence_counts_befor_after_pre_processing.png', dpi=300)


neg_sentences = None
pos_sentences = None
neu_sentences = None
with open('data/sentencebroken/neg_data_cleaned.txt', 'r') as f:
  neg_sentences = np.array(f.read().split('\n'))

with open('data/sentencebroken/pos_data_cleaned.txt', 'r') as f:
  pos_sentences = np.array(f.read().split('\n'))

with open('data/sentencebroken/neu_data_cleaned.txt', 'r') as f:
  neu_sentences = np.array(f.read().split('\n'))

print("number of negative sentences: ", len(neg_sentences))
print("number of positive sentences: ", len(pos_sentences))
print("number of neutral sentences: ", len(neu_sentences))

labels = ['Negative sentences', 'Positive sentences', 'Neutral sentences']
values = [len(neg_sentences), len(pos_sentences), len(neu_sentences)]

#save in csv file
tmp_df = pd.DataFrame()
tmp_df['neg_sentences_num'] = [neg_sentences]
tmp_df['pos_sentences_num'] = [pos_sentences]
tmp_df['neu_sentences_num'] = [neu_sentences]
tmp_df.to_csv('data/stats/sentences_for_each_label.csv')

fig, ax = plt.subplots(figsize=(8, 6))

# Create a bar chart
ax.bar(labels, values, width=0.3)

# Set the y-axis to have 5 steps
ax.yaxis.set_major_locator(MaxNLocator(6))
ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

# Customize the chart
ax.set_ylabel('sentence counts')
ax.set_title('sentences')

# Display the chart
fig.savefig('data/stats/sentences_for_each_label.png', dpi=300)

neg_tokens = None
pos_tokens = None
neu_tokens = None
with open('data/wordbroken/neg_data_cleaned.txt', 'r') as f:
  neg_tokens = np.array(f.read().split(' '))

with open('data/wordbroken/pos_data_cleaned.txt', 'r') as f:
  pos_tokens = np.array(f.read().split(' '))

with open('data/wordbroken/neu_data_cleaned.txt', 'r') as f:
  neu_tokens = np.array(f.read().split(' '))

print("number of negative tokens: ", len(neg_tokens))
print("number of positive tokens: ", len(pos_tokens))
print("number of neutral tokens: ", len(neu_tokens))

labels = ['Negative tokens', 'Positive tokens', 'Neutral tokens']
values = [len(neg_tokens), len(pos_tokens), len(neu_tokens)]

#save in csv file
tmp_df = pd.DataFrame()
tmp_df['neg_tokens_num'] = [neg_tokens]
tmp_df['pos_tokens_num'] = [pos_tokens]
tmp_df['neu_tokens_num'] = [neu_tokens]
tmp_df.to_csv('data/stats/tokens_for_each_label.csv')

fig, ax = plt.subplots(figsize=(8, 6))

# Create a bar chart
ax.bar(labels, values, width=0.3)

# Set the y-axis to have 5 steps
ax.yaxis.set_major_locator(MaxNLocator(6))
ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

# Customize the chart
ax.set_ylabel('token counts')
ax.set_title('tokens')

# Display the chart
fig.savefig('data/stats/tokens_for_each_label.png', dpi=300)


distinct_neg_tokens = set(neg_tokens)
distinct_pos_tokens = set(pos_tokens)
distinct_neu_tokens = set(neu_tokens)

print("number of distinct negative tokens: ", len(distinct_neg_tokens))
print("number of distinct positive tokens: ", len(distinct_pos_tokens))
print("number of distinct neutral tokens: ", len(distinct_neu_tokens))

labels = ['Negative tokens', 'Positive tokens', 'Neutral tokens']
values = [len(distinct_neg_tokens), len(distinct_pos_tokens), len(distinct_neu_tokens)]

#save in csv file
tmp_df = pd.DataFrame()
tmp_df['distinct_neg_tokens_num'] = [distinct_neg_tokens]
tmp_df['distinct_pos_tokens_num'] = [distinct_pos_tokens]
tmp_df['distinct_neu_tokens_num'] = [distinct_neu_tokens]
tmp_df.to_csv('data/stats/distinct_tokens_for_each_label.csv')

fig, ax = plt.subplots(figsize=(8, 6))

# Create a bar chart
ax.bar(labels, values, width=0.3)

# Set the y-axis to have 5 steps
ax.yaxis.set_major_locator(MaxNLocator(6))
ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

# Customize the chart
ax.set_ylabel('token counts')
ax.set_title('Distinct tokens')

# Display the chart
fig.savefig('data/stats/distinct_tokens_for_each_label.png', dpi=300)

shared_distinct_tokens = distinct_neg_tokens.intersection(distinct_pos_tokens, distinct_neu_tokens)

unique_distinct_neg_tokens = distinct_neg_tokens.difference(distinct_pos_tokens, distinct_neu_tokens)
unique_distinct_pos_tokens = distinct_pos_tokens.difference(distinct_neg_tokens, distinct_neu_tokens)
unique_distinct_neu_tokens = distinct_neu_tokens.difference(distinct_neg_tokens, distinct_pos_tokens)

print("Number of shared distinct tokens:", len(shared_distinct_tokens))
print("number of unique  distinct negative tokens: ", len(unique_distinct_neg_tokens))
print("number of unique  distinct positive tokens: ", len(unique_distinct_pos_tokens))
print("number of unique  distinct neutral tokens: ", len(unique_distinct_neu_tokens))

labels = ['Shared distinct tokens', 'unique Negative tokens', 'unique Positive tokens', 'unique Neutral tokens']
values = [len(shared_distinct_tokens), len(unique_distinct_neg_tokens), len(unique_distinct_pos_tokens), len(unique_distinct_neu_tokens)]

#save in csv file
tmp_df = pd.DataFrame()
tmp_df['shared_distinct_tokens_num'] = [shared_distinct_tokens]
tmp_df['unique_distinct_neg_tokens_num'] = [unique_distinct_neg_tokens]
tmp_df['unique_distinct_pos_tokens_num'] = [unique_distinct_pos_tokens]
tmp_df['unique_distinct_neu_tokens_num'] = [unique_distinct_neu_tokens]
tmp_df.to_csv('data/stats/shared_&_unique_distinct_tokens_for_each_label.csv')

fig, ax = plt.subplots(figsize=(8, 6))

# Create a bar chart
ax.bar(labels, values, width=0.3)

# Set the y-axis to have 5 steps
ax.yaxis.set_major_locator(MaxNLocator(6))
ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

# Customize the chart
ax.set_ylabel('token counts')
ax.set_title('shared & unique tokens')

# Display the chart
fig.savefig('data/stats/shared_&_unique_distinct_tokens_for_each_label.png', dpi=300)

import collections

# Count the frequency of each token in each set
neg_token_counts = collections.Counter(neg_tokens)
pos_token_counts = collections.Counter(pos_tokens)
neu_token_counts = collections.Counter(neu_tokens)

# Get the top-10 most common tokens for each set
top_neg_tokens = [(token, count) for token, count in neg_token_counts.most_common() if token in unique_distinct_neg_tokens][:10]
top_pos_tokens = [(token, count) for token, count in pos_token_counts.most_common() if token in unique_distinct_pos_tokens][:10]
top_neu_tokens = [(token, count) for token, count in neu_token_counts.most_common() if token in unique_distinct_neu_tokens][:10]

# Print the results
print("Top-10 most repeated tokens for negative sentiment:")
for token, count in top_neg_tokens:
    print(f"{token}: {count}")

print()
print("Top-10 most repeated tokens for positive sentiment:")
for token, count in top_pos_tokens:
    print(f"{token}: {count}")

print()
print("Top-10 most repeated tokens for neutral sentiment:")
for token, count in top_neu_tokens:
    print(f"{token}: {count}")

# Write the top-10 most repeated tokens for each sentiment to separate CSV files
for label, top_tokens in zip(['Negative', 'Positive', 'Neutral'], [top_neg_tokens, top_pos_tokens, top_neu_tokens]):
    df = pd.DataFrame(top_tokens, columns=['Token', 'Count'])
    df.to_csv(f'data/stats/top_10_most_repeated_tokens_{label.lower()}.csv', index=False)
    


# Set font size for x-axis tick labels
plt.rcParams['xtick.labelsize'] = 60

# Top negative tokens
tokens, counts = zip(*top_neg_tokens)
plt.bar(tokens, count, width=0.2)
plt.xticks(rotation=45, ha='right')
plt.xlabel('Negative Token', fontsize=40)
plt.ylabel('Count', fontsize=40)
plt.yticks(fontsize=40)
plt.yticks(range(0, max(counts)+1, 5))
plt.title('Top Negative Tokens', fontsize=40)
plt.savefig('data/stats/top_10_most_repeated_tokens_negative.png')

# Top positive tokens
tokens, counts = zip(*top_pos_tokens)
plt.bar(tokens, counts, width=0.2)
plt.xticks(rotation=45, ha='right')
plt.xlabel('Positive Token', fontsize=14)
plt.ylabel('Count', fontsize=40)
plt.yticks(fontsize=40)
plt.yticks(range(0, max(counts)+1, 5))
plt.title('Top Positive Tokens', fontsize=40)
plt.savefig('data/stats/top_10_most_repeated_tokens_positive.png')

# Top neutral tokens
tokens, counts = zip(*top_neu_tokens)
plt.bar(tokens, counts, width=0.2)
plt.xticks(rotation=45, ha='right')
plt.xlabel('Neutral Token', fontsize=40)
plt.ylabel('Count', fontsize=40)
plt.yticks(fontsize=40)
plt.yticks(range(0, max(counts)+1, 5))
plt.title('Top Neutral Tokens', fontsize=40)
plt.savefig('data/stats/top_10_most_repeated_tokens_neutral.png')


# 1. find the TF-IDF of all tweets.
def compute_tf_idf(tweets, unique_words):
  unique_word_size = len(unique_words)
  n_docs = len(tweets)
  tf = np.zeros((n_docs, unique_word_size))
  tf_idf = np.zeros((n_docs, unique_word_size))
  idf = np.zeros(unique_word_size)
  number_of_doc_contain_word = np.zeros(unique_word_size)

  # Compute Term Frequency (TF)
  for i in range(n_docs):
      words = tokenizer.tokenize(tweets[i]) # Words in the document
      for w in words:
          tf[i][unique_words.index(w)] = tf[i][unique_words.index(w)] + (1 / len(words))
      for w in set(words):
        number_of_doc_contain_word[unique_words.index(w)] += 1

  #compute IDF
  for i,w in enumerate(unique_words):
    idf[i] = np.log10(n_docs/number_of_doc_contain_word[i])

  #compute tf-idf
  for i in range(n_docs):
    words = tokenizer.tokenize(tweets[i])
    for w in words:
      tf_idf[i][unique_words.index(w)] = tf[i][unique_words.index(w)] * idf[unique_words.index(w)]

  return tf_idf


from sklearn.feature_extraction.text import TfidfVectorizer

# Combine the cleaned tweets for each label into a single string.tolist()
neg_tweets = clean_df[clean_df['sentiment'] == -1]['tweet_cleaned'].tolist()
pos_tweets = clean_df[clean_df['sentiment'] == 1]['tweet_cleaned'].tolist()
neu_tweets = clean_df[clean_df['sentiment'] == 0]['tweet_cleaned'].tolist()

# Create a TfidfVectorizer object and fit it to the tweets for each label
vectorizer = TfidfVectorizer()
neg_tfidf = vectorizer.fit_transform(neg_tweets)
pos_tfidf = vectorizer.fit_transform(pos_tweets)
neu_tfidf = vectorizer.fit_transform(neu_tweets)

# Get the indices of the top-10 tokens for each label
neg_top_indices = neg_tfidf.toarray()[0].argsort()[-10:][::-1]
pos_top_indices = pos_tfidf.toarray()[0].argsort()[-10:][::-1]
neu_top_indices = neu_tfidf.toarray()[0].argsort()[-10:][::-1]

# Get the feature names for the top-10 tokens for each label
feature_names = vectorizer.get_feature_names_out()

# Retrieve the top tokens for each label
neg_top_tokens = [feature_names[i] for i in neg_top_indices if i < len(feature_names)]
pos_top_tokens = [feature_names[i] for i in pos_top_indices if i < len(feature_names)]
neu_top_tokens = [feature_names[i] for i in neu_top_indices if i < len(feature_names)]

# Print the results
print("Top-10 tokens for negative sentiment based on TF-IDF:")
print(neg_top_tokens)
print("\nTop-10 tokens for positive sentiment based on TF-IDF:")
print(pos_top_tokens)
print("\nTop-10 tokens for neutral sentiment based on TF-IDF:")
print(neu_top_tokens)

# Write the top-10 tokens based on TF-IDF for each sentiment to separate CSV files
for label, top_tokens in zip(['Negative', 'Positive', 'Neutral'], [top_neg_tokens, top_pos_tokens, top_neu_tokens]):
    df = pd.DataFrame(top_tokens, columns=['Token', 'Count'])
    df.to_csv(f'data/stats/top_10_tokens_tf_idf_{label.lower()}.csv', index=False)
    
    
    
# Create a pandas Series of the token counts
token_counts = pd.Series(cleaned_tokens).value_counts()

# Select the top 100 tokens with the highest frequency
top_tokens = token_counts.nlargest(100).sort_values(ascending=True)

# Plot a horizontal bar chart of the top tokens
fig, ax = plt.subplots()
top_tokens.plot(kind='barh', ax=ax)
ax.set_xlabel('Token Count')
ax.set_ylabel('Token')
ax.set_title('Top 100 Tokens with Highest Frequency')

# Display the token counts under each bar
for i, v in enumerate(top_tokens):
    ax.text(v + 1, i, str(v), color='blue', fontweight='bold')

fig.savefig('data/stats/top_100_high_frequency_tokens_histogram.png', dpi=300)


# Sort the tokens in descending order of frequency
sorted_tokens = token_counts.sort_values(ascending=False)

# Plot a histogram of the token counts
fig, ax = plt.subplots()
ax.hist(range(len(sorted_tokens)), weights=sorted_tokens, bins=100)
ax.set_xlabel('Token')
ax.set_ylabel('Count')
ax.set_title('Histogram of All Tokens in Descending Order')

# Save the plot to a PNG file
fig.savefig('data/stats/histogram_of_all_tokens.png', dpi=300)