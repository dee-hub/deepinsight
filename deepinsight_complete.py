from nltk.corpus import stopwords
from textblob import TextBlob
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import nltk
nltk.download('vader_lexicon')
import re
import string
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
import meta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import snscrape.modules.twitter as sntwitter
import requests
import time  # to simulate a real time data, time loop
import plotly.express as px
import streamlit as st  # 🎈 data web app development

st.set_page_config(
    page_title="Tweet Analysis",
    page_icon="logo.png",
    layout="wide",
)

st.title('DeepInsight 📊v3.0 ')
st.markdown("<i>Unleash the power of Twitter with DeepInsight - where every tweet tells a story.</i>", unsafe_allow_html=True)
selected = st.text_input("Enter a keyword to analyze")
from_date = st.date_input("Select the start date", datetime.date(2023, 4, 1))
to_date = st.date_input("Select the end date")
length = st.radio("Number of tweets to retrieve", (10, 50, 100, 250, 500, 1000, 2500, 5000))
button_analyze = st.button('Extract Tweets')
# creating a single-element container
placeholder = st.empty()
    
#@st.cache
def clean_text(keyword, date_start, date_end, tweet_length):
    import re
    tweets_list3 = []
    import time
    st.markdown("Extracting tweet data related to " + "\'" + str(keyword) + "\'")
    my_bar = st.progress(0)
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(keyword + ' since:'+ str(date_start) + ' until:' + str(date_end)).get_items()):
        if i>tweet_length:
            break
        tweets_list3.append([tweet.date, tweet.id, tweet.content, tweet.user.username, tweet.likeCount, tweet.retweetCount, tweet.replyCount, tweet.user.location])
        my_bar.progress((i/tweet_length))
        print('Tweet ' + str(i) + ' appended')
    tweets = pd.DataFrame(tweets_list3, columns=['Datetime', 'Tweet Id', 'Text', 'Username', 'likes', 'retweets', 'replies', 'location'])
    tweets['engagements'] = tweets['likes'] + tweets['retweets'] + tweets['replies']
    rt = lambda x: re.sub("(@(?!BAT|Tinubushettima|PeterObi|obidatti|Kwankwanso|Atiku)[A-Za-z0-9_]+)|(https?:\/\/[^\s]+)|([^\x00-\x7F])|(\n)|(\",)", " ", x)
    tweets['Text'] = tweets.Text.map(rt)
    tweets['Text'] = tweets['Text'].map(lambda x: re.sub(' +', ' ', x))
    tweets['Text'] = tweets.Text.str.lower()
    tweets['Text'] = tweets['Text'].str.replace('dey play o', 'Just keep playing')
    return tweets

def sentiment_analysis(df):
    analyzer = SentimentIntensityAnalyzer()
    df['sentiment'] = df['Text'].apply(lambda x: analyzer.polarity_scores(x))
    df['neg'] = df['sentiment'].apply(lambda x: x['neg'])
    df['neu'] = df['sentiment'].apply(lambda x: x['neu'])
    df['pos'] = df['sentiment'].apply(lambda x: x['pos'])
    df['compound'] = df['sentiment'].apply(lambda x: x['compound'])
    df['sentiment'] = df['compound'].apply(lambda x: 'positive' if x >= 0 else 'negative')
    sentiment_analysis_data = df.copy()
    return sentiment_analysis_data

def plot_sentiments(df):
    sentiment_data = sentiment_analysis(df)
    pos_count = sentiment_data[sentiment_data['sentiment'] == 'positive']['sentiment'].count()
    neg_count = sentiment_data[sentiment_data['sentiment'] == 'negative']['sentiment'].count()
    labels = ['Positive', 'Negative']
    sizes = [pos_count, neg_count]
    explode = (0.05, 0.05)
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, explode=explode, pctdistance=0.85)
    centre_circle = plt.Circle((0, 0), 0.40, fc='white')
    fig1 = plt.gcf()
    fig1.gca().add_artist(centre_circle)
    ax1.axis('equal') 
    return fig1

def plot_engagements(data):
    df = sentiment_analysis(data)    
    tweet_df = df.sort_values(by=['engagements'], ascending=False)
    top_tweets = tweet_df[['Datetime', 'Username', 'Text', 'engagements', 'sentiment', 'location']].head(10)
    return top_tweets

def plot_location(data):
    df = plot_engagements(data)
    latitude = []
    longitude = []
    for location in df['location']:
        url = f'https://nominatim.openstreetmap.org/search?q={location}&format=json'
        response = requests.get(url, timeout=50).json()
        if len(response) > 0:
            latitude.append(response[0]['lat'])
            longitude.append(response[0]['lon'])
        else:
            latitude.append(None)
            longitude.append(None)
    df['latitude'] = latitude
    df['longitude'] = longitude
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df = df.dropna(subset=['latitude', 'longitude'])
    fig = px.scatter_mapbox(df, lat='latitude', lon='longitude', color='location',
                            hover_name='location', hover_data=['Username', 'engagements'],
                            zoom=2, height=600,
                            title='Top Engaged Tweets by Location')
    # Update the mapbox style
    #fig.update_layout(mapbox_style='open-street-map')
    fig.update_layout(
    mapbox_style="white-bg",
    mapbox_layers=[
        {
            "below": 'traces',
            "sourcetype": "raster",
            "sourceattribution": "United States Geological Survey",
            "source": [
                "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
            ]
        },
        {
            "sourcetype": "raster",
            "sourceattribution": "Government of Canada",
            "source": ["https://geo.weather.gc.ca/geomet/?"
                       "SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&BBOX={bbox-epsg-3857}&CRS=EPSG:3857"
                       "&WIDTH=1000&HEIGHT=1000&LAYERS=RADAR_1KM_RDBR&TILED=true&FORMAT=image/png"],
        }
      ])
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    return st.plotly_chart(fig, use_container_width=True)

def word_cloud(df):
    sentiment_data_cloud = sentiment_analysis(df)
    stop_words = set(stopwords.words("english"))
    stop_words.update(["the", "your", "ur", "want", "what", "wat", "said", "try", "go"])
    sentiment_data_cloud['Text'] = sentiment_data_cloud['Text'].str.replace('[{}]'.format(string.punctuation), '')
    sentiment_data_cloud['Text'] = sentiment_data_cloud['Text'].str.replace(r'\b\w{1,3}\b', '')
    mask = np.array(Image.open('cloud.png'))
    wordcloud = WordCloud(background_color='white',
                   mask = mask,
                   max_words=20,
                   stopwords=stop_words,
                   repeat=False)
    text = sentiment_data_cloud['Text'].values
    wordcloud.generate(str(text))
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    return fig

if selected and from_date and to_date and length and button_analyze:
    dataframe = clean_text(selected, from_date, to_date, length)
    if length <= 1000:
        fig_col1, fig_col2 = st.columns(2)
        with fig_col1:
            st.markdown("Sentiment Analysis")
            st.write(plot_sentiments(dataframe))
        with fig_col2:
            st.markdown("Prominent Words in Tweet")
            st.write(word_cloud(dataframe))
        try:
            st.write(plot_engagements(dataframe))
            plot_location(dataframe)
        except:
            st.markdown('Map unavailable')
        
        #st.dataframe(sentiment_analysis(dataframe)[['Text', 'sentiment']].tail())
    elif length > 1000:
        st.markdown("<p> Contact the developer directly <a href='https://linkedin.com/in/oluwadolapo-salako'>here</a> for analysis of upto 10 million tweets </p>", unsafe_allow_html=True)