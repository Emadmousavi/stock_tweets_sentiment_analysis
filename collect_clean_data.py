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

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

stop_words = stopwords.words('english')

sia = SentimentIntensityAnalyzer()


df = pd.read_csv("data/raw/raw_stock_tweet_sentiment.csv")

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

print('--- cleaning process started -----')

df['tweet'] = df['tweet'].astype('str')
df['tweet_cleaned'] = df['tweet'].apply(preprocess)

os.makedirs('data/clean/', exist_ok=True)
df.to_csv("data/clean/clean_stock_tweet_sentiment_1.csv")