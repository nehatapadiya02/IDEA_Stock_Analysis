import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import json
import requests
from datetime import date, timedelta
from newsapi import NewsApiClient
from helper import parse_sentiment_scores, predict_trend, preprocess_news,predict_sentiments


ticker_keyword = [
    ("SBIN.NS", "sbi"),
    ("HDFCBANK.NS", "hdfc"),
    ("RELIANCE.NS", "reliance"),
    ("ADANIENT.NS", "adani"),
    ("ZOMATO.NS", "zomato"),
    ("DMART.NS", "dmart"),
    ("IRCTC.NS", "irctc"),
    ("ITC.NS", "itc"),
    ("TECHM.NS", "techm"),
    ("NTPCGREEN.NS", "ntpc green"),
    ("TCS.NS", "tcs"),
    ("LICI.NS", "lic"),
    ("VEDL.NS", "vedanta"),
    ("HYUNDAI.NS", "hyundai"),
    ("ONGC.NS", "ongc"),
    ("TITAN.NS", "titan"),
    ("ASIANPAINT.NS", "asian paints"),
    ("AUROPHARMA.NS", "aurobindo pharma"),
    ("LUPIN.NS", "lupin"),
    ("PAYTM.NS", "paytm"),
    ("INFY.NS", "infosys"),
    ("TATAPOWER.NS", "tata power"),
    ("STARHEALTH.NS", "star health"),
    ("COCHINSHIP.NS", "cochin shipyard")
]

hindi_ticker_keyword = [
    ("SBIN.NS", "भारतीय स्टेट बैंक"),
    ("HDFCBANK.NS", "एचडीएफसी बैंक"),
    ("RELIANCE.NS", "रिलायंस इंडस्ट्रीज"),
    ("ADANIENT.NS", "अदानी"),
    ("ZOMATO.NS", "ज़ोमैटो"),
    ("DMART.NS", "डीमार्ट"),
    ("IRCTC.NS", "आईआरसीटीसी"),
    ("ITC.NS", "आईटीसी"),
    ("TECHM.NS", "टेक महिंद्रा"),
    ("NTPCGREEN.NS", "एनटीपीसी ग्रीन"),
    ("TCS.NS", "टीसीएस"),
    ("LICI.NS", "भारतीय जीवन बीमा निगम"),
    ("VEDL.NS", "वेदांता"),
    ("HYUNDAI.NS", "हुंडई"),
    ("ONGC.NS", "ओएनजीसी"),
    ("TITAN.NS", "टाइटन"),
    ("ASIANPAINT.NS", "एशियन पेंट्स"),
    ("AUROPHARMA.NS", "औरोबिंदो फार्मा"),
    ("LUPIN.NS", "ल्यूपिन"),
    ("PAYTM.NS", "पेटीएम"),
    ("INFY.NS", "इंफोसिस"),
    ("TATAPOWER.NS", "टाटा पावर"),
    ("STARHEALTH.NS", "स्टार हेल्थ"),
    ("COCHINSHIP.NS", "कोचीन शिपयार्ड")
]

ticker_dict = dict(ticker_keyword)
hindi_ticker_dict = dict(hindi_ticker_keyword)

# Constants
# NEWS_API_KEY = "838016d46b6145bd8228dee046f86671"
NEWS_API_KEY = "3c72746100464eb99e5242d3f9f79aab"
newsapi = NewsApiClient(api_key=NEWS_API_KEY)
BASE_URL = "https://newsapi.org/v2/everything"

# Sidebar Layout
st.sidebar.title('IDEA - Stock Analysis')

def reset_rerun_flag():
    st.session_state.rerun_triggered = False

selected_ticker = st.sidebar.selectbox(
    "Select a Stock",
    list(ticker_dict.keys()),
    on_change = reset_rerun_flag
)

st.sidebar.write("**Selected Stock:**", selected_ticker)

# Date Selection
st.sidebar.subheader("Select Date Range")
today = date.today()
from_date = st.sidebar.date_input("From", today - timedelta(days=365))
to_date = st.sidebar.date_input("To", today - timedelta(days=0))

if "predict_actual_row_present" not in st.session_state:
    st.session_state.predict_actual_row_present = None

recommendation_map = {
    0: "Hold ➖",
    1: "Buy 📈",
    2: "Sell 📉"
}

action_colors = {
    1: "#2ecc71",   # Green
    2: "#e74c3c",  # Red
    0: "#f39c12"   # Amber/Orange
}


if st.session_state.predict_actual_row_present:
    predicted_row = st.session_state.predicted_price_row
    original_row = st.session_state.original_price_row
    st.markdown(
        f"""
        <div style="text-align: center; font-size: 20px; font-weight: bold; padding: 10px; border-radius: 10px; background-color: black; color: white;">
            <p>💰 <b>Actual Price:</b> <span style="color: #1f77b4;">{original_row['Close']}</span></p>
            <p>📉 <b>Predicted Price:</b> <span style="color: #ff7f0e;">{predicted_row['Trend_Regression']}</span></p>
            <p>📢 <b>Predicted Action:</b> <span style="color: {action_colors.get(predicted_row['Trend_Classification'])};">{recommendation_map.get(predicted_row['Trend_Classification'])}</span></p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Fetch Stock Data
stock = yf.Ticker(selected_ticker)
data = stock.history(start=from_date, end=to_date)
data = data.reset_index()
data.rename(columns={'index': 'Date'}, inplace=True)
print(f"Loaded price data {data.info()}")

data["Date"] = pd.to_datetime(data["Date"].dt.date)
fig = px.line(data, x='Date', y='Close', title=f"{stock.info.get('shortName')}")
st.plotly_chart(fig, use_container_width=True)

if "predicted_data_present" not in st.session_state:
    st.session_state.predicted_data_present = None

if "rerun_triggered" not in st.session_state:
    st.session_state.rerun_triggered = False 

if st.session_state.predicted_data_present:
    data_with_trend = st.session_state.predicted_data
    # fig_classification = px.line(data_with_trend, x='Adjusted_Date', y='Trend_Classification', title=f"Predicted Stock Trend {stock.info.get('shortName')}")
    fig_regression = px.line(data_with_trend, x='Adjusted_Date', y='Trend_Regression', title=f"Predicted Stock Trend {stock.info.get('shortName')}")
    st.plotly_chart(fig_regression, use_container_width=True)
    # st.plotly_chart(fig_classification, use_container_width=True)


# Tabbed Interface
price_data, fundamental_data, news = st.tabs(["📈 Price Data", "📊 Fundamental Data", "📰 News"])

with price_data:
    st.subheader("Price Data and Performance Metrics")
    data_copy = data.copy()
    data_copy['% Change'] = data_copy['Close'].pct_change()
    data_copy.dropna(inplace=True)
    annual_returns = data_copy['% Change'].mean() * 252 * 100
    st_dev = np.std(data_copy['% Change']) * np.sqrt(252) * 100
    risk_adjusted_return = annual_returns / st_dev if st_dev != 0 else np.nan
    
    # Display Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Annual Returns", f"{annual_returns:.2f}%")
    col2.metric("Standard Deviation", f"{st_dev:.2f}%")
    col3.metric("Risk Adjusted Return", f"{risk_adjusted_return:.2f}")
    
    st.dataframe(data_copy, use_container_width=True)

with fundamental_data:
    st.subheader("Fundamental Stock Information")
    st.markdown(f"**Company Summary:** {stock.info.get('longBusinessSummary')}")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Market Cap", f"{stock.info.get('marketCap'):,}")
    col2.metric("PE Ratio", f"{stock.info.get('trailingPE')}")
    col3.metric("PB Ratio", f"{stock.info.get('priceToBook')}")
    
    col4, col5 = st.columns(2)
    col4.metric("52-Week High", f"{stock.info.get('fiftyTwoWeekHigh')}")
    col5.metric("52-Week Low", f"{stock.info.get('fiftyTwoWeekLow')}")

    st.json(stock.info, expanded=False)

with news:
    st.subheader("Latest News")
    keyword = ticker_dict.get(selected_ticker)
    news_list = []
    articles = newsapi.get_everything(
        qintitle=keyword,
        from_param=today - timedelta(days=25),
        to=today,
        language='en',
        sort_by='publishedAt',
        page_size=10
    )
    hindi_keyword = hindi_ticker_dict.get(selected_ticker)
    params = {
        "qintitle": hindi_keyword,
        "from_param": today - timedelta(days=25),
        "to":today,
        "language": "hi",
        "sortBy": "publishedAt",
        "pageSize": 10,
        "apiKey": NEWS_API_KEY,
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        hindi_articles = response.json()["articles"]
    else:
        print(f"Error fetching news for {selected_ticker}: {response.status_code}, {response.text}")

    for article in hindi_articles:
        news_list.append({
            'Date': article['publishedAt'][:10],  # Extract date
            'Title': article['title'],
            'Description': article['description'],
            'URL': article['url']
        })
        st.markdown(f"**{article['title']}**")
        st.write(article['description'])
        st.write(article['publishedAt'][:10]),
        st.write(f"[Read more]({article['url']})")
        st.write("---")
    
    for article in articles.get('articles', []):
        news_list.append({
            'Date': article['publishedAt'][:10],  # Extract date
            'Title': article['title'],
            'Description': article['description'],
            'URL': article['url']
        })
        st.markdown(f"**{article['title']}**")
        st.write(article['description'])
        st.write(article['publishedAt'][:10]),
        st.write(f"[Read more]({article['url']})")
        st.write("---")
    
    news_data = pd.DataFrame(news_list)
    news_data["Date"] = pd.to_datetime(news_data["Date"])
    news_data["Date_Formated"] = pd.to_datetime(news_data["Date"].dt.date)
    news_data["Title"] = news_data["Title"].str.lower().str.replace(r"[^\w\s]", "", regex=True)
    news_data["Description"] = news_data["Description"].str.lower().str.replace(r"[^\w\s]", "", regex=True)
    news_data['full_news'] = ' ---title--- ' + news_data['Title'].astype(str) + ' ---body--- ' + news_data['Description'].astype(str) + ' ---newarticle--- '
    news_data['news_combined'] = news_data.groupby(['Date_Formated'])['full_news'].transform(lambda x: ' '.join(x))
    news_data['news_combined_processed'] = news_data['news_combined'].apply(preprocess_news)
    news_data['sentiment_scores'] = news_data['news_combined_processed'].apply(predict_sentiments)
    news_data['sentiment_scores_parsed'] = news_data['sentiment_scores'].apply(parse_sentiment_scores)
    sentiment_columns = ['negative_score', 'neutral_score', 'positive_score']
    sentiment_df = pd.DataFrame(news_data['sentiment_scores_parsed'].to_list(), columns=sentiment_columns)
    news_data = pd.concat([news_data, sentiment_df], axis=1)
    news_data.drop(columns=['sentiment_scores','sentiment_scores_parsed'], inplace=True)

    print(f"Price Data: {data.info()}")
    print(f"News Data: {news_data.info()}")
    data_with_trend = predict_trend(price_data =data,news_data = news_data)
    st.session_state.predicted_data = data_with_trend
    max_date_predicted_row = data_with_trend.loc[data_with_trend["Adjusted_Date"].idxmax()]
    max_date_original_row = data.loc[data["Adjusted_Date"].idxmax()]
    st.session_state.original_price_row = max_date_original_row
    st.session_state.predicted_price_row = max_date_predicted_row
    st.session_state.predicted_data_present = True
    st.session_state.predict_actual_row_present = True

    if not st.session_state.rerun_triggered:
        st.session_state.rerun_triggered = True
        st.rerun()        
    # st.write(f"Predicted Trend: {data_with_trend}")
    print(f"data with trend {data_with_trend.head(10)}")
    fig_classification = px.line(data_with_trend, x='Adjusted_Date', y='Trend_Classification', title="Predicted Stock Trend")
    fig_regression = px.line(data_with_trend, x='Adjusted_Date', y='Trend_Regression', title="Predicted Stock Trend")
    st.plotly_chart(fig_classification, use_container_width=True)
    st.plotly_chart(fig_regression, use_container_width=True)
    # st.dataframe(news_data, use_container_width=True)