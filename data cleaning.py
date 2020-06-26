### clean twitter data ###
import pandas as pd
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


# load tweets from all jsons
pd.set_option('display.float_format', lambda x: '%.f' % x)

# Read json into a pandas dataframe
output_dir = "C:/Users/leoba/PycharmProjects/sicss2020/research project/raw data/"
tweets_df = pd.read_json(output_dir + "20quat_alradae-tweets.json", lines=True)
tweets_df2 = pd.read_json(output_dir + "604Sj66-tweets.json", lines=True)
tweets_df3 = pd.read_json(output_dir + "aliabozguia1-tweets.json", lines=True)
tweets_df4 = pd.read_json(output_dir + "anshangol-tweets.json", lines=True)
tweets_df5 = pd.read_json(output_dir + "blresh92-tweets.json", lines=True)
tweets_df6 = pd.read_json(output_dir + "boshgma-tweets.json", lines=True)
tweets_df7 = pd.read_json(output_dir + "dodycat42-tweets.json", lines=True)
tweets_df8 = pd.read_json(output_dir + "efta_libya-tweets.json", lines=True)
tweets_df9 = pd.read_json(output_dir + "hamoalgali-tweets.json", lines=True)
tweets_df10 = pd.read_json(output_dir + "LiBya_73-tweets.json", lines=True)
tweets_df11 = pd.read_json(output_dir + "mohamedmerabet7-tweets.json", lines=True)
tweets_df12 = pd.read_json(output_dir + "muatez_elfarsi-tweets.json", lines=True)
tweets_df13 = pd.read_json(output_dir + "tarekdorman-tweets.json", lines=True)
tweets_df14 = pd.read_json(output_dir + "TawfekElkmate-tweets.json", lines=True)
tweets_df15 = pd.read_json(output_dir + "Yo8sF-tweets.json", lines=True)
# glue them together
mad_final = pd.concat([tweets_df, tweets_df2, tweets_df3, tweets_df4, tweets_df5, tweets_df6, tweets_df7,
                       tweets_df8, tweets_df9, tweets_df10, tweets_df11, tweets_df12, tweets_df13, tweets_df14,
                       tweets_df15])


# save as csv
mad_final.to_csv('mad_full.csv')

# load from csv
mad_final_csv = pd.read_csv('research project\\raw data\\mad_full.csv')

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
punctuations = '''!()-–[]{};:'"„“\,<>./?@#€$%^&*_~'''
string3 = ""
for char in string2:
   if char not in punctuations:
       string3 = string3 + char
print(string3)
len(string3)

# remove latin characters
latin = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
string4 = ""
for char in string3:
    if char not in latin:
        string4 = string4 + char
print(string4)
len(string4)

# Tokenize
words = nltk.word_tokenize(string4)
print(words)
len(words)

# Normalize
words = normalize(words)
print(words)
len(words)

# fix for matplotlib to display allah
res = [sub.replace('الله', 'ا لله') for sub in words]

# stem words (a disaster with arabic)
words_stm = stem_words(words)
print(words_stm)

# most frequent words
count_all = Counter()
count_all.update(res)
print(count_all.most_common(10))

# first plot (horizontal bar graph)
wordsdf = pd.DataFrame(count_all.most_common(10), columns=['Words', 'Count'])

x = []

for item in wordsdf.Words.values:
    x.append(get_display(arabic_reshaper.reshape(item)))


fig, ax = plt.subplots(figsize=(8, 8))
wordsdf.sort_values(by='Count').plot.barh(x='Words',
                      y='Count',
                      ax=ax,
                      color="green")
ax.set_yticklabels(reversed(x))
ax.set_title("Top 10 Unigrams Found in Tweets (No Stop Words)")

plt.tight_layout()
plt.show()
plt.savefig('top_10_unigrams_no_stop_words.png')

# create bigrams (paired adjacent terms in a tuple)
bigrams = ngrams(res,2)

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
                      color="green")
ax.set_yticklabels(reversed(x))
ax.set_title("Top 10 Bigrams Found in Tweets (No Stop Words)")

plt.tight_layout()
plt.show()
plt.savefig('top_10_bigrams_no_stop_words.png')

# trigrams
trigrams = ngrams(res,3)

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
                      color="green")
ax.set_yticklabels(reversed(x))
ax.set_title("Top 10 Trigrams Found in Tweets (No Stop Words)")

plt.tight_layout()
plt.show()
plt.savefig('top_10_trigrams_no_stop_words.png')
