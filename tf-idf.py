### TF IDF ###
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction import text
import json
import http.client
from sklearn.cluster import KMeans


# df = pd.read_csv('research project\\raw data\\mad_full.csv', low_memory=False)

pd.set_option('display.float_format', lambda x: '%.f' % x)

output_dir = "research project/raw data/"

df = pd.read_json(output_dir + "20quat_alradae-tweets.json", lines=True)

df.to_csv('rada_full.csv')

df = pd.read_csv('research project\\raw data\\rada_full.csv')


# build arrays with tweet text and tweet id
X = df.iloc[:, 4].values
y = df.iloc[:, 2].values

# clean tweets
processed_tweets = []

for tweet in range(0, len(X)):
    # Remove all the special characters
    processed_tweet = re.sub(r'\W', ' ', str(X[tweet]))

    # remove all single characters
    processed_tweet = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_tweet)

    # remove urls
    processed_tweet = re.sub(r'http\S+', ' ', processed_tweet)

    # remove digits
    processed_tweet = re.sub(r'\d+', ' ', processed_tweet)

    # Remove single characters from the start
    processed_tweet = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_tweet)

    # Substituting multiple spaces with single space
    processed_tweet = re.sub(r'\s+', ' ', processed_tweet, flags=re.I)

    # Removing prefixed 'b'
    processed_tweet = re.sub(r'^b\s+', '', processed_tweet)

    processed_tweets.append(processed_tweet)


print(processed_tweets)

# kick out latin characters and other stuff
no_latin = str.maketrans("", "", "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZäöüÄÖÜ_এℹআâছΜыঘধইতযপখİḤжьТজНхôГгèবরÇчȘМডমюДকдـīבяßםつくсনづлркпйтиıунваāğçéоем")

ptnew = [s.translate(no_latin) for s in processed_tweets]
print(ptnew)

# kick out extra spaces
ptreg = []

for tweet in range(0, len(ptnew)):
    ptfresh = re.sub(r'\s+', ' ', str(ptnew[tweet]))

    ptfresh = re.sub(r'^\s+|\s+$', '', ptfresh)

    ptreg.append(ptfresh)


print(ptreg)

# check for non arabic characters and iterate the no_latin equation
ptstr = str(ptreg)

counter = Counter(ptstr)
print(counter)

# create document term matrix
count_vectorizer = CountVectorizer()
ptmat = count_vectorizer.fit_transform(ptreg)

ptmat

ptmat.toarray()

pd.DataFrame(ptmat.toarray())

# add column headers
count_vectorizer.get_feature_names()

# assign column headers
matrix1 = pd.DataFrame(ptmat.toarray(), columns=count_vectorizer.get_feature_names())

# remove stop words
with open('stopwords-arabic.txt', 'r', encoding='utf8') as f:
    stopwords = f.read().splitlines() # splitlines(): create a list using space

stop_words = text.ENGLISH_STOP_WORDS.union(stopwords)

count_vectorizer = CountVectorizer(stop_words=stop_words)

xx = count_vectorizer.fit_transform(ptreg)
print(count_vectorizer.get_feature_names())

# assign new column headers
matrix2 = pd.DataFrame(xx.toarray(), columns=count_vectorizer.get_feature_names())

# term frequency
tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, use_idf=False)
x5 = tfidf_vectorizer.fit_transform(ptreg)
matrix3 = pd.DataFrame(x5.toarray(), columns=tfidf_vectorizer.get_feature_names())

# inversed term frequency
idf_vectorizer = TfidfVectorizer(stop_words=stop_words, use_idf=True)
x6 = idf_vectorizer.fit_transform(ptreg)
idf_df = pd.DataFrame(x6.toarray(), columns=idf_vectorizer.get_feature_names())
idf_df

# search for words
# freak = pd.DataFrame([idf_df['قرنهم']], index=["قرنهم"]).T

# topical clusters
vectorizer = TfidfVectorizer(use_idf=True, stop_words=stop_words)
x7 = vectorizer.fit_transform(ptreg)

number_of_clusters = 5
km = KMeans(n_clusters=number_of_clusters)
km.fit(x7)

# print clustered words
print("Top terms per cluster:")
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(number_of_clusters):
    top_ten_words = [terms[ind] for ind in order_centroids[i, :5]]
    print("Cluster {}: {}".format(i, ' '.join(top_ten_words)))


km.labels_

# create dataframe with cluster-no. assigned to documents
results = pd.DataFrame()
results['text'] = ptreg
results['category'] = km.labels_
results

# create smaller cluster to plot
vectorizer = TfidfVectorizer(use_idf=True, max_features=2, stop_words=stop_words)
keta = vectorizer.fit_transform(ptreg)
vectorizer.get_feature_names()

dfSDF = pd.DataFrame(keta.toarray(), columns=vectorizer.get_feature_names())
dfSDF


dfSDFcopy = dfSDF.rename(columns={'الله': 'God', 'قوةالردعلخاصة': 'SDF'})

ax = dfSDF.plot(kind='scatter', x='الله', y='قوةالردعالخاصة', alpha=0.1, s=300)
ax.set_xlabel("God")
ax.set_ylabel("Special Deterrence Force")
ax.set_title("")
