import streamlit as st
import pandas as pd
import os
import random
from re import search
from PIL import Image
from spacy.lang.en.stop_words import STOP_WORDS
stopwords = list(STOP_WORDS)
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import seaborn as sns
import plotly.express as px
import urllib.request
import matplotlib.pyplot as plt
import time
import os
import time
#from stqdm import stqdm
from textblob import TextBlob
from bs4 import BeautifulSoup as bs
import requests
if 'data' not in st.session_state:
    st.session_state.data = 0
if 'options' not in st.session_state:
    st.session_state.options = None
info_dict = {"Camera": ['photos', 'pictures', 'camera'],
             "Performance": ["fast", "processor", "processing", "speed", "performance"],
             "Software": ["bugs", "ui", "software"], "Battery": ["battery", "charge", "charging", "long use"],
             "Display": ["Display", "Screen", "hz"]}
path = "https://raw.githubusercontent.com/adityajoe/MobilePhone_Review_Summarization/main/Project_Dashboard/pages"
def getPolarity(text):
    return TextBlob(text).sentiment.polarity
def getAnalysis2(polarity):
    if polarity > 0:
        return "Positive"
    else:
        return "Negative"
def getAnalysis(df):
    classes = []
    for i in df.index:
        if df["Title_Polarity"][i] > 0 and df["Review_Polarity"][i] >0:
            classes.append("Positive")
        else:
            classes.append("Negative")
    df["Class"] = classes
    return df

def get_data(link):
    my_bar = st.progress(0)
    new_reviews = []
    new_ratings = []
    new_titles = []
    percentage_complete = 0
    #link = link
    for count in range(1, 51):
        #st.write(i,percentage_complete)
        newlink = link[0:len(link) - 1] + str(count)
        page = requests.get(newlink)
        soup = bs(page.content, 'html.parser')
        soup.prettify()
        reviews = soup.find_all("div", class_="t-ZTKy")
        ratings = soup.find_all("div", class_="_3LWZlK _1BLPMq")
        titles = soup.find_all("p", class_="_2-N8zT")
        for i in range(0, len(reviews)):
            l = len(reviews[i].get_text())
            new_titles.append(titles[i].get_text())
            new_reviews.append(reviews[i].get_text()[:l - 9])
        for i in range(0, len(ratings)):
            new_ratings.append(ratings[i].get_text())
        percentage_complete += 2
        if count == 50:
            percentage_complete = 100
        my_bar.progress(percentage_complete)
    st.write("Scraping Completed!")
    product_reviews = pd.DataFrame(
        {
            "Review Title": new_titles,
            "Review Text": new_reviews
        })
    product_reviews["Title_Polarity"] = product_reviews["Review Title"].apply(getPolarity)
    product_reviews["Review_Polarity"] = product_reviews["Review Text"].apply(getPolarity)
    product_reviews = getAnalysis(product_reviews)
    return product_reviews
    # print(len(new_reviews))
    # print(len(new_ratings))
    # print(len(new_titles))

def n_grams():
    c_vec = CountVectorizer(stop_words=stopwords, ngram_range=(2, 3))
    # matrix of ngrams
    ngrams = c_vec.fit_transform(st.session_state.data['Review Text'])
    # count frequency of ngrams
    count_values = ngrams.toarray().sum(axis=0)
    # list of ngrams
    vocab = c_vec.vocabulary_
    df_ngram = pd.DataFrame(sorted([(count_values[i], k) for k, i in vocab.items()], reverse=True)
                            ).rename(columns={0: 'frequency', 1: 'bigram/trigram'})
    text = list(df_ngram['bigram/trigram'][:15])
    return text


def analytics():
    st.title("Scraped Dataset: " + name)
    st.caption("PS. Hover over a review to read it completely")
    st.dataframe(st.session_state.data)
    st.write("")
    st.write("")
    reviews = st.session_state.data["Review Text"]
    titles = st.session_state.data["Review Title"]
    # st.header(header + ": Analytics")
    positive = st.session_state.data["Class"][st.session_state.data["Class"] == "Positive"].count()
    negative = st.session_state.data["Class"][st.session_state.data["Class"] == "Negative"].count()
    # rating = st.session_state.data["Score"].mean()
    Total = st.session_state.data.shape[0]
    st.title("Basic Statistics")
    st.subheader("Total Number of Reviews: " + str(Total))
    st.subheader("Positive Reviews: " + str(positive))
    st.subheader("Negative Reviews: " + str(negative))
    st.subheader("Likeability Factor: " + str(np.round(float(positive/Total), 2)))
    # st.subheader("Average Rating: " + str(rating * 2) + "/10")

    if st.button("Display Top Reviews"):
        st.caption("Please Click button to reload reviews")
        titles = []
        reviews = []
        # st.header("Top Reviews!")
        for i in range(5):
            num = random.randint(0, 20)
            titles.append(st.session_state.data["Review Title"].iloc[num])
            reviews.append(st.session_state.data["Review Text"].iloc[num])
        for i in range(5):
            st.header(" " + titles[i])
            st.write(reviews[i])
    st.header("Filter Reviews by Keywords")
    num_reviews = len(reviews)
    string = "Given below are the most occurring phrases in all the " + str(
        num_reviews) + " mobile reviews present for this model"
    st.caption(string)
    col1, col2 = st.columns(2)
    result = list(n_grams())
    # result = result[:7]
    dict1 = {}

    with col1:
        for i in result:
            st.code(i)
    with col2:
        st.caption("You can search for reviews with keywords here by copy pasting them here!")
        keyword = st.text_input("Search By Keyword!(eg. camera, battery life, software)")
        count = 0
        random.shuffle(reviews)
        keyword = keyword.split()
        for review in reviews:
            if any(substring in review for substring in keyword):
                count += 1
                title = np.array(st.session_state.data["Review Title"][st.session_state.data["Review Text"] == review])
                title = str(title[0])
                # sst.write(str(title).split())
                st.subheader(title)
                st.write(str(count) + ". " + review)
                st.write("")
            if count == 3:
                count = 0
                break

def aspects():
    text = np.array(st.session_state.data["Review Text"])
    sentences = []
    new_sentences = []
    for i in text:
        sentence = i.split('.')
        sentences.extend(sentence)
    for sentence in sentences:
        if len(sentence) <= 3:
            sentences.remove(sentence)
    for sentence in sentences:
        if 'but' in sentence:
            new_sentences.extend(sentence.split('but'))
        elif 'although' in sentence:
            new_sentences.extend(sentence.split('although'))
        else:
            new_sentences.append(sentence)
    details = {'sentence_reviews': new_sentences
               }
    df_reviews = pd.DataFrame(details)
    df_reviews["Review_Polarity"] = df_reviews["sentence_reviews"].apply(getPolarity)
    df_reviews["Class"] = df_reviews["Review_Polarity"].apply(getAnalysis2)
    text_sent = df_reviews['sentence_reviews']
    text_sent = np.unique(np.array(text_sent))
    key_topics = []
    likes = []
    dislikes = []
    st.session_state.options = st.multiselect(
        'Summarize using your favorite aspects',
        ['Camera', 'Battery', 'Performance', 'Software', 'Display'])
    for option in st.session_state.options:
        all = info_dict[option]
        positive = 0
        negative = 0
        count = 0
        for word in all:
            for i in text_sent:
                if search(word.lower(), str(i).lower()):
                    a = np.array(df_reviews["Class"][df_reviews['sentence_reviews'] == i])[0]
                    if a == 'Positive':
                        positive += 1
                    else:
                        negative += 1
                    count += 1
        col1, col2 = st.columns(2)
        with col1:
            st.write("*************************")
            st.header(option + "!")
            if (positive / count) * 100 > 70:
                str1 = "1. This phone has very good reviews for " + option + "!" + " If you are a ' " + option + "-centric  \
             person', You should go for this!"
            else:
                str1 = "1. This phone has very bad reviews for " + option + "!" + " If you are a ' " + option + "-centric " \
                                                                                                                "person', You should not go for this "
            st.write()
            st.write(str1)
            str2 = "Out of a total of " + str(count) + " people, "
            st.subheader(str2)
            str3 = "Positive: " + str(positive)
            str4 = "Negative: " + str(negative)
            st.subheader(str3)
            st.subheader(str4)
            key_topics.append(option)
            likes.append((positive / count) * 100)
            dislikes.append((negative / count) * 100)
        with col2:
            labels = ["Positive Reviews", "Negative Reviews"]
            arr = [(positive / count) * 100, (negative / count) * 100]
            explode = (0.1, 0)
            fig1, ax1 = plt.subplots()
            ax1.pie(arr, explode=explode, labels=labels, autopct='%1.1f%%',
                    shadow=True, startangle=90)
            ax1.axis('equal')
            st.pyplot(fig1)  # use_container_width=True)
            st.write()
            st.write()
    st.header("Main features and their Likeabailtiy Factors")
    main_topics = pd.DataFrame(list(zip(key_topics, likes, dislikes)),
                               columns=['Key Features', 'Likeability', 'Percentage_dislike'])
    fig = px.bar(main_topics, x='Key Features', y='Likeability', color="Key Features")
    st.plotly_chart(fig)  # , use_container_width=True)

#main
st.title("Analyse Your Phone in Realtime!")
a = urllib.request.urlretrieve(os.path.join(path,"image.jpg"))
image = Image.open(a[0])
st.image(image)
#header = st.text_input("Enter the name of your Phone!")
link = st.text_input("Enter flipkart link of the product")
link = link + "&page=1"
name = ""
for char in link[25:50]:
    if char != "/":
        name += char
    else:
        break
st.caption("""Choose any mobile phone on flipkart, go to reviews section and click on page 2 of reviews.
           Paste that link in the box above""")
if st.button("Start Scraping"):
    st.session_state.data = get_data(link)
    st.session_state.flag = 1
st.write("")
st.write("")
try:
    analytics()
except:
    pass
try:
    aspects()
except:
    pass

