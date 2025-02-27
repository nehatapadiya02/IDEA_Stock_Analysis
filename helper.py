import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pickle
import pandas as pd
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# sentiment_model = BertForSequenceClassification.from_pretrained("./trained_model")
sentiment_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
sentiment_model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=3)


def best_model_classification():
    with open('best_model_classification.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    return loaded_model

classification_model = best_model_classification()

def best_model_regression():
    with open('best_model_regression.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    return loaded_model

regression_model = best_model_regression()    

def load_hindi_stopwords(filepath):
    """Loads Hindi stopwords from a text file."""
    with open(filepath, 'r', encoding='utf-8') as file:
        stopwords = set(file.read().splitlines())
    return stopwords

hindi_stop_words = load_hindi_stopwords('stopwords-hi.txt')
english_stop_words = set(stopwords.words('english'))

stop_words = hindi_stop_words.union(english_stop_words)

def preprocess_news(news):
  # news = re.sub(r'[^a-zA-Z]', ' ',news)
  # news = news.lower()
  # tokens = word_tokenize(news)
    news = re.sub(r'[^a-zA-Z\u0900-\u097F\s]', ' ', news)

    # Convert text to lowercase (affects only English characters; Hindi remains unaffected)
    news = news.lower()

    # Tokenize the text into words
    tokens = word_tokenize(news)

  # stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)


def predict_sentiments(news):
  inputs = sentiment_tokenizer(news, return_tensors="pt", truncation=True, padding=True, max_length = 256)
  outputs = sentiment_model(**inputs)
  probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
  return probs.detach().numpy()


def parse_sentiment_scores(score_string):
  """Parses a sentiment score string and returns a list of floats."""
  score_string = str(score_string).strip('[]')  # Remove brackets
  scores = score_string.split()  # Split by whitespace
  return [float(score) for score in scores]  # Convert to floats


def predict_stock_classification(data):
    return classification_model.predict(data)

def predict_stock_regression(data):
    return regression_model.predict(data)


def next_business_day(date, business_dates):
    while date not in business_dates:
        date += pd.Timedelta(days=1)
    return date

def predict_trend(price_data , news_data):
    
    print(f"Price Data: {price_data.info()}")
    print(f"News data raw: {news_data.head(10)}")
    price_data["Adjusted_Date"] = pd.to_datetime(price_data["Date"].dt.date)
    price_data.sort_values(by=["Adjusted_Date"])
    business_dates = set(price_data["Adjusted_Date"])
    news_data["Adjusted_Date"] = news_data["Date_Formated"].apply(
    lambda x: next_business_day(x, business_dates) if x.weekday() >= 5 else x
    )
    print(f"News data : {news_data.head(10)}")
    # Combine price and news data
    price_data.reset_index(drop=True)
    news_data.reset_index(drop=True)
    merged_data =  pd.merge(price_data, news_data, on=["Adjusted_Date"], how="right")
    merged_data.dropna(inplace=True)
    merged_data.sort_values(by=["Adjusted_Date"], inplace=True)
    print(f"merged data : {merged_data.info()}")
    columns_to_keep_classification = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'negative_score', 'neutral_score', 'positive_score']
    columns_to_keep_regression = ['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits', 'negative_score', 'neutral_score', 'positive_score']
    final_data_classification = merged_data[columns_to_keep_classification]
    final_data_regression = merged_data[columns_to_keep_regression]
    
    print(f"final data classification : {final_data_classification.info()}")
    # Predict the stock trend
    trend_classification = predict_stock_classification(final_data_classification)
    print(f"Predicted Trend: {trend_classification}")
    merged_data["Trend_Classification"] = trend_classification

    print(f"final data regression: {final_data_regression.info()}")
    # Predict the future price
    trend_regression = predict_stock_regression(final_data_regression)
    print(f"Predicted Trend: {trend_regression}")
    merged_data["Trend_Regression"] = trend_regression

    return merged_data

