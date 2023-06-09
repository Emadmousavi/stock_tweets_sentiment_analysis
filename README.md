# stock_tweets_sentiment_analysis


### Dataset links:
##### 1- https://www.kaggle.com/datasets/chenxidong/stock-tweet-sentiment-dataset
##### 2- https://huggingface.co/datasets/mjw/stock_market_tweets
##### 3- https://www.kaggle.com/datasets/andreaskonradsen/stocks-with-sentiment-and-emotion-analysis
##### 4- https://www.kaggle.com/datasets/fabioturazzi/cryptocurrency-tweets-with-sentiment-analysis


### My Dataset link (HuggingFace):
##### https://huggingface.co/datasets/emad12/stock_tweets_sentiment/viewer/emad12--stock_tweets_sentiment/

<br>
<br>


### How to use:
#### in order to collect data and stats successfully you can use NLP_Project.ipynb notebook. <br>
#### Alternatively you can run this python file in sequence to have same output as notebbok:
##### 1- collect raw data
```
python collect_raw_data.py
```

##### 1- collect clean data
```
python collect_clean_data.py
```

##### 1- tokenize data
```
python tokenize_data.py
```

##### 1- collect stats
```
python collect_stats.py
```

#### or you can just run this command
```
python collect_data_&_stats.py
```
