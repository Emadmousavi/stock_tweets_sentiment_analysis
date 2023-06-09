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

print('-------- collecting raw data started ------')

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