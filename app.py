#pip install pyspark
#pip install streamlit

import streamlit as st
import requests
import pandas as pd
import altair as alt
from textblob import TextBlob
#from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
import datetime

#Select Ticker
ticker=st.selectbox('Select ticker',('AAPL','MSFT','AMZN','TSLA','GOOG','FB','TWTR'))

#Select Date 
#today = datetime.date.today()
#tomorrow = today + datetime.timedelta(days=1)
start_date = st.date_input('Start date', value=pd.to_datetime('2022-05-02'))
end_date = st.date_input('End date', value=pd.to_datetime('2022-05-05'))
if start_date > end_date:
    st.error('Error: End date must fall after start date.')


#Establishing URL parameters
url = f"https://api.twelvedata.com/time_series?start_date={start_date}&end_date={end_date}&symbol={ticker}&interval=1day&apikey=f2230c31479b4c02bd2c86d80e5a3beb&source=docs"


#Retrieving data from Website
r = requests.get(url)
result = r.json()
#st.json(result)
aapl = result['values']
df=pd.DataFrame(aapl)

y_data_max = df['close'].max()
y_data_min = df['close'].min()

#Create line graph 
line_chart = alt.Chart(df).mark_line().encode(
  x=alt.X('datetime:N'),
  y=alt.Y('close:Q', scale=alt.Scale(domain=(y_data_min, y_data_max )))
).properties(title=f"{ticker} Close Price") #Agregar Titulo dependiendo del valor de ticker
st.altair_chart(line_chart, use_container_width=True)



#Retrieve News Data 

#Set up NEWS API 

r_news = requests.get(f"https://eodhistoricaldata.com/api/news?api_token=62684169e4c281.91841636&s={ticker}.US&from={start_date}&to={end_date}")
result_news = r_news.json()
#print (type(result_news).__name__)
#print(result_news)

df = pd.DataFrame(result_news)


no_articles=len(df.index)
st.metric(label="Number of articles analyzed", value=no_articles)

from sklearn.feature_extraction.text import CountVectorizer


#Polarity - Text not clean
df['polarity'] = df['title'].apply(lambda tweet: TextBlob(tweet).sentiment.polarity)
polarity=df['polarity'].mean()
st.metric(label="Title Polarity Index", value=polarity)

#Sentiment - Text not clean
def getAnalysis(score):
  if score < 0:
    return 'Negative'
  elif score == 0:
    return 'Neutral'
  else:
    return 'Positive'
df['TextBlob_Analysis'] =df['polarity'].apply(getAnalysis )


import nltk
nltk.download('punkt')
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,ENGLISH_STOP_WORDS


import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import collections
import tweepy as tw
from nltk.corpus import stopwords
import re
import numpy as np

##ANALISIS DE CONTENIDO DE LOS ARTICULOS
#Pasar columna a lista 
list_of_content = df['content'].tolist()
#print(list_of_content)

#Remover Símbolos
def remove_url(txt):

    return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", txt).split())

#Generar lista sin símbolos
all_no_urls = [remove_url(signos) for signos in list_of_content]

##Generar lista en minúsculas, juntar todos los diccionarios 
words = [article.lower().split() for article in all_no_urls]
all_words_no_urls = list(itertools.chain(*words))


from nltk.corpus import stopwords
nltk.download('stopwords')

#Determinar stopwords
stop_words = set(stopwords.words('english'))

##Remover stopwords de words 
for all_words in words:
    for word in all_words:
        tweets_nsw = [[word for word in tweet_words if not word in stop_words]
              for tweet_words in words]

#Crear lista collection_words
collection_words = ['inc', 'apple', '2022','shares','aapl','stock','2021','also','may','price','million','billion','company','stocks','markets','year','analyst','year','quarter','value']

#Quitar las collection words 
tweets_nsw_nc = [[w for w in word if not w in collection_words]
                 for word in tweets_nsw]


# Flatten list of words in clean tweets
all_words_nsw_nc = list(itertools.chain(*tweets_nsw_nc))

# Create counter of words in clean tweets
counts_nsw_nc = collections.Counter(all_words_nsw_nc)

#Pasar a DatFrame
clean_tweets_ncw = pd.DataFrame(counts_nsw_nc.most_common(15),
                             columns=['words', 'count'])


#Crear plot final
fig, ax = plt.subplots(figsize=(8, 8))

# Plot horizontal bar graph
clean_tweets_ncw.sort_values(by='count').plot.barh(x='words',
                      y='count',
                      ax=ax,
                      color="purple")

ax.set_title("Common Words Found in Articles")

plt.show()
st.pyplot(fig)

####ANALISIS DE TITULO DE LOS ARTICULOS
#Pasar columna a lista 
list_of_content = df['title'].tolist()


#Remover Símbolos
def remove_url(txt):

    return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", txt).split())

#Generar lista sin símbolos
all_no_urls = [remove_url(signos) for signos in list_of_content]

##Generar lista en minúsculas, juntar todos los diccionarios 
words = [article.lower().split() for article in all_no_urls]
all_words_no_urls = list(itertools.chain(*words))


from nltk.corpus import stopwords
nltk.download('stopwords')

#Determinar stopwords
stop_words = set(stopwords.words('english'))

##Remover stopwords de words 
for all_words in words:
    for word in all_words:
        tweets_nsw = [[word for word in tweet_words if not word in stop_words]
              for tweet_words in words]

#Crear lista collection_words
collection_words = ['inc', 'apple', '2022','shares','aapl','stock','2021','also','may','price','million','billion','company','stocks','markets','year','analyst','year','quarter','value']

#Quitar las collection words 
tweets_nsw_nc = [[w for w in word if not w in collection_words]
                 for word in tweets_nsw]


# Flatten list of words in clean tweets
all_words_nsw_nc = list(itertools.chain(*tweets_nsw_nc))

# Create counter of words in clean tweets
counts_nsw_nc = collections.Counter(all_words_nsw_nc)

#Pasar a DatFrame
clean_tweets_ncw = pd.DataFrame(counts_nsw_nc.most_common(15),
                             columns=['words', 'count'])


#Crear plot final
fig_title, ax = plt.subplots(figsize=(8, 8))

# Plot horizontal bar graph
clean_tweets_ncw.sort_values(by='count').plot.barh(x='words',
                      y='count',
                      ax=ax,
                      color="purple")

ax.set_title("Common Words Found in Titles")

plt.show()
st.pyplot(fig_title)



###BORRADOR

def getAnalysis(score):
  if score < 0:
    return 'Negative'
  elif score == 0:
    return 'Neutral'
  else:
    return 'Positive'
df['TextBlob_Analysis'] =df['polarity'].apply(getAnalysis )

sentiment_score=df['polarity'].mean()




