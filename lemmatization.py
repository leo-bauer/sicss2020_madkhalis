### lemmatization ###
import pandas as pd
import numpy as np
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.isri import ISRIStemmer
import matplotlib.pyplot as plt
from bidi.algorithm import get_display
import arabic_reshaper
from nltk.util import ngrams
import json
import http.client
from string import digits
from wordcloud import WordCloud


# load from csv
mad_final_csv = pd.read_csv('research project\\raw data\\mad_full.csv', low_memory=False)

# count tweets
len(mad_final_csv)
# count retweets
mad_final_csv['retweeted_status'].count()
# start cleaning by creating a list
text = mad_final_csv.full_text.tolist()
print(text)
# create string from list
textstr = str(text)
print(textstr)
### let the cleanup begin ###

def remove_URL(textstr):
    """Remove URLs from a sample string"""
    return re.sub(r"http\S+", "", textstr)


with open('stopwords-arabic.txt', 'r', encoding='utf8') as f:
    stopwords= f.read().splitlines() # splitlines(): create a list using space
stopwords= set(stopwords)

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords:
            new_words.append(word)
    return new_words


def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = ISRIStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems


def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas


def normalize(words):
    words = remove_stopwords(words)
    return words

# HERE THE ACTUAL WORK BEGINS
# remove urls
string1 = remove_URL(textstr)
print(string1)

# remove digits
string2 = ''.join(i for i in string1 if not i.isdigit())
print(string2)
len(string2)

# remove punctuation
punctuations = '''!()-–[]{};:…''️'"„“\,<>./؟?@#€♥$%^&*_~'''
string3 = ""
for char in string2:
   if char not in punctuations:
       string3 = string3 + char
print(string3)
len(string3)

# remove arabic punctuations
intab = '?'
outtab = ' '
trantab = str.maketrans(intab, outtab)
string4 = string3.translate(trantab)

# remove latin characters
latin = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
string5 = ""
for char in string4:
    if char not in latin:
        string5 = string5 + char
print(string5)
len(string5)

# lemmatization using Farasa API
conn = http.client.HTTPSConnection("farasa-api.qcri.org")
payload = "{\"text\":\"%s\"}"% string5
payload = payload.encode('utf-8')

headers = { 'content-type': "application/json", 'cache-control': "no-cache", }

conn.request("POST", "/msa/webapi/lemma", payload, headers)

res = conn.getresponse()

data = res.read().decode('utf-8')
data_dict = json.loads(data)

# convert results from dict to list
list11 = data_dict['result']
print(list11)
len(list11)

# remove stop words
words_lem = normalize(list11)
print(words_lem)
len(words_lem)

# fix for matplotlib to display allah (WHAT A GIANT MESS!!! WHAT IS THIS SORCERY??? WHY WON'T THE FU$%&NG '?' FUCK OFF???)
res = [sub.replace('الله', 'ا لله') for sub in words_lem]
res1 = [sub.replace('?', '-') for sub in res]
res2 = [s for s in res1 if s != '-']

# count most frequent words
count_all = Counter()
count_all.update(res2)
print(count_all.most_common(10))

# plot unigrams
wordsdf = pd.DataFrame(count_all.most_common(10), columns=['Words', 'Count'])

x = []

for item in wordsdf.Words.values:
    x.append(get_display(arabic_reshaper.reshape(item)))


fig, ax = plt.subplots(figsize=(8, 8))
wordsdf.sort_values(by='Count').plot.barh(x='Words',
                      y='Count',
                      ax=ax,
                      color="red")
ax.set_yticklabels(reversed(x))
ax.set_title("Top 10 Unigrams Found in Tweets (Lemmatized)")

plt.tight_layout()
plt.show()
plt.savefig('top_10_unigrams_lemmatized.png')

# create bigrams
bigrams = ngrams(res2,2)

# convert to list
blist = list(bigrams)
print(blist)

# sort bigrams
count_all = Counter()
count_all.update(blist)
count_all.most_common(11)

# create DataFrame from bigram list
dfb = pd.DataFrame(blist, columns=['bigram1', 'bigram2'])
dffreq = dfb.groupby(["bigram1", "bigram2"]).size().reset_index(name="Count")
# create DataFrame with 20 most frequent bigrams
viz = dffreq.sort_values(by=["Count"], ascending=False)[1:11]
# merge string columns and clean DataFrame
viz['Bigrams'] = viz.bigram1.str.cat(viz.bigram2,sep=" ")
viz = viz.drop(['bigram1', 'bigram2'], axis=1)
viz = viz[['Bigrams', 'Count']]

# bigram plot with horizontal bar graph
x = []

for item in viz.Bigrams.values:
    x.append(get_display(arabic_reshaper.reshape(item)))


fig, ax = plt.subplots(figsize=(8, 8))
viz.sort_values(by='Count').plot.barh(x='Bigrams',
                      y='Count',
                      ax=ax,
                      color="red")
ax.set_yticklabels(reversed(x))
ax.set_title("Top 10 Bigrams Found in Tweets (Lemmatized)")

plt.tight_layout()
plt.show()
plt.savefig('top_10_bigrams_lemmatized.png')

# trigrams
trigrams = ngrams(res2,3)

# convert to list
tlist = list(trigrams)

# sort trigrams
count_all = Counter()
count_all.update(tlist)
count_all.most_common(11)

# create DataFrame from trigram list
dft = pd.DataFrame(tlist, columns=['trigram1', 'trigram2', 'trigram3'])
dffreq = dft.groupby(["trigram1", "trigram2", 'trigram3']).size().reset_index(name="Count")
# create DataFrame with 20 most frequent bigrams
viz = dffreq.sort_values(by=["Count"], ascending=False)[1:11]
# merge string columns and clean DataFrame
viz['Trigrams'] = viz.trigram1.str.cat(viz.trigram2,sep=" ")
viz['Trigrams'] = viz.Trigrams.str.cat(viz.trigram3,sep=" ")
viz = viz.drop(['trigram1', 'trigram2'], axis=1)
viz = viz[['Trigrams', 'Count']]

# trigram plot with horizontal bar graph
x = []

for item in viz.Trigrams.values:
    x.append(get_display(arabic_reshaper.reshape(item)))


fig, ax = plt.subplots(figsize=(8, 8))
viz.sort_values(by='Count').plot.barh(x='Trigrams',
                      y='Count',
                      ax=ax,
                      color="red")
ax.set_yticklabels(reversed(x))
ax.set_title("Top 10 Trigrams Found in Tweets (Lemmatized)")

plt.tight_layout()
plt.show()
plt.savefig('top_10_trigrams_lemmatized.png')
