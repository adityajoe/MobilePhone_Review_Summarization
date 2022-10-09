import streamlit as st
import pandas as pd
import os
import random
from re import search
from PIL import Image
from spacy.lang.en.stop_words import STOP_WORDS
stopwords = list(STOP_WORDS)
from sklearn.feature_extraction.text import CountVectorizer
model_name = ["poco", "x3", "pro"]
raw_path = "https://raw.githubusercontent.com/adityajoe/MobilePhone_Review_Summarization/main/Project_Dashboard"
stopwords.extend(model_name)
import numpy as np
import seaborn as sns
import plotly.express as px
import urllib.request
import matplotlib.pyplot as plt
import time
import os
def get_data(path):
    path1 = os.path.join(path,"labelled_reviews.csv")
    path2 = os.path.join(path,"single_aspect_reviews.csv")
    df = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    return df, df2

def n_grams():
    c_vec = CountVectorizer(stop_words=stopwords, ngram_range=(2, 3))
    # matrix of ngrams
    ngrams = c_vec.fit_transform(data['Review Text'])
    # count frequency of ngrams
    count_values = ngrams.toarray().sum(axis=0)
    # list of ngrams
    vocab = c_vec.vocabulary_
    df_ngram = pd.DataFrame(sorted([(count_values[i], k) for k, i in vocab.items()], reverse=True)
                            ).rename(columns={0: 'frequency', 1: 'bigram/trigram'})
    text = list(df_ngram['bigram/trigram'][:15])
    return text


def aspects():
    key_topics = []
    likes = []
    dislikes = []
    options = st.multiselect(
        'Summarize using your favorite aspects',
        ['Camera', 'Battery', 'Performance', 'Software', 'Gaming', 'Display'])
    for option in options:
        col1, col2 = st.columns(2)
        positive = 0
        negative = 0
        count = 0
        with col1:
            st.write("*************************")
            st.header(option + "!")
            for i in text_sent:
                if search(option.lower(), str(i).lower()):
                    a = np.array(df_reviews["Class"][df_reviews['sentence_reviews'] == i])[0]
                    if a == 'Positive':
                        positive += 1
                    else:
                        negative += 1
                    count += 1
            if (positive / count) * 100 > 60:
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
            explode = (0.1,0)
            fig1, ax1 = plt.subplots()
            ax1.pie(arr, explode=explode, labels=labels, autopct='%1.1f%%',
                    shadow=True, startangle=90)
            ax1.axis('equal')
            st.pyplot(fig1)# use_container_width=True)
            st.write()
            st.write()
    st.header("Main features and their Likeabailtiy Factors")
    main_topics = pd.DataFrame(list(zip(key_topics, likes, dislikes)),
                               columns=['Key Features', 'Likeability', 'Percentage_dislike'])
    fig = px.bar(main_topics, x='Key Features', y='Likeability', color="Key Features")
    st.plotly_chart(fig) #, use_container_width=True)

st.title("Mobile Phone Review Summarization")
mobile = st.selectbox("Select the Phone Type",
                      ("Poco X3 Pro", "Iphone SE"))
if mobile == "Poco X3 Pro":
    path = os.path.join(raw_path,"poco")
    header = "POCO X3 PRO!"
elif mobile == "Iphone SE":
    path = os.path.join(raw_path,"apple")
    header = "Iphone SE"
a = urllib.request.urlretrieve(os.path.join(path,"image.jpg"))
image = Image.open(a[0])
data, df_reviews = get_data(path)
reviews = data["Review Text"]
titles = data["Review Title"]
st.header(header)
st.image(image)
positive = data["Class"][data["Class"] == "Positive"].count()
negative = data["Class"][data["Class"] == "Negative"].count()
rating = data["Score"].mean()
Total = data.shape[0]
st.subheader("Total Number of Reviews: " + str(Total))
st.subheader("Positive Reviews: " +  str(positive))
st.subheader("Negative Reviews:" + str(negative))
st.subheader("Average Rating: " + str(rating * 2) + "/10")

if st.button("Display Top Reviews"):
    st.caption("Please Click button to reload reviews")
    titles = []
    reviews = []
    # st.header("Top Reviews!")
    for i in range(5):
        num = random.randint(0,200)
        titles.append(data["Review Title"].iloc[num])
        reviews.append(data["Review Text"].iloc[num])
    for i in range(5):
        st.header(" " + titles[i])
        st.write(reviews[i])
st.header("Filter Reviews by Keywords")
num_reviews = len(reviews)
string = "Given below are the most occurring phrases in all the " + str(num_reviews) + " mobile reviews present for this model"
st.caption(string)
col1, col2 = st.columns(2)
result = list(n_grams())
#result = result[:7]
dict1 = {}

with col1:
    for i in result:
        st.code(i)
with col2:
    st.caption("You can search for reviews with keywords here by copy pasting them here!")
    keyword = st.text_input("Search By Keyword!(eg. camera, battery life, software")
    count  = 0
    random.shuffle(reviews)
    keyword = keyword.split()
    for review in reviews:
        if any(substring in review for substring in keyword):
            count += 1
            title = np.array(data["Review Title"][data["Review Text"] == review])
            title = str(title[0])
            #sst.write(str(title).split())
            st.subheader(title)
            st.write(str(count) + ". " +  review)
            st.write("")
        if count == 3:
            count = 0
            break

st.write()
st.write()
st.header("Aspect Based Sentiment Analysis")
st.caption("Here, you can find the sentiment analysis according to every"
           " aspect of the phone! For eg. What people feel about the camera, what people feel about the processor etc")

text_sent = df_reviews['sentence_reviews']
classes = df_reviews['Class']
aspects()
#text_sent = np.unique(np.array(text_sent))

    

