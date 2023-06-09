# stock_tweets_sentiment_analysis
## About Project
#### This is IUST NLP Course project. I have done these items in this project:
- ##### Use some different Kaggle Datasets about Stock and Crypto tweets, combine them, pre-process them and finally create a new Dataset
- ##### Label each tweets as positive, negative and neutral with some auto-labling techniques
- ##### show Statistics about Dataset in PNG and CSV formats
- ##### Create a Transformer-based model and train it on my dataset

#### Finally you can feed model a tweet about Stock or crypto and get its sentiment as positve, negtivae or neutral
#### You can read the projcet document [HERE](https://github.com/Emadmousavi/stock_tweets_sentiment_analysis/blob/main/latex_report/main.pdf)
<br>

## Used-Datasets links:
##### 1- https://www.kaggle.com/datasets/chenxidong/stock-tweet-sentiment-dataset
##### 2- https://huggingface.co/datasets/mjw/stock_market_tweets
##### 3- https://www.kaggle.com/datasets/andreaskonradsen/stocks-with-sentiment-and-emotion-analysis
##### 4- https://www.kaggle.com/datasets/fabioturazzi/cryptocurrency-tweets-with-sentiment-analysis

<br>

## My Dataset link (HuggingFace):
##### https://huggingface.co/datasets/emad12/stock_tweets_sentiment/viewer/emad12--stock_tweets_sentiment/

<br>

## How to use:
#### In order to collect data and stats successfully you can use NLP_Project.ipynb notebook. <br>
#### Alternatively you can run this python file in sequence to have same output as notebbok:
##### 1- collect raw data
```
python collect_raw_data.py
```

##### 2- collect clean data
```
python collect_clean_data.py
```

##### 3- tokenize data
```
python tokenize_data.py
```

##### 4- collect stats
```
python collect_stats.py
```
### or you can just run this command
```
python collect_data_&_stats.py
```
