import logging

import pandas

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from spacy.lang.ru import Russian
from spacy_russian_tokenizer import RussianTokenizer, MERGE_PATTERNS

parser = Russian()
russian_tokenizer = RussianTokenizer(parser, MERGE_PATTERNS)
parser.add_pipe(russian_tokenizer, name='russian_tokenizer')


def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    print([token.text for token in tokens])
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens


import nltk

nltk.download('wordnet')
from nltk.corpus import wordnet as wn


def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma


from nltk.stem.wordnet import WordNetLemmatizer


def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)


nltk.download('stopwords')
ru_stop = set(nltk.corpus.stopwords.words('russian'))


def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 3]
    tokens = [token for token in tokens if token not in ru_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens


import random

text_data = []
df = pandas.read_csv("reviews.csv", encoding="utf-8")
reviews = df['text']
print(len(reviews))

for review in reviews:
    print(review)
    tokens = prepare_text_for_lda(review)
    if random.random() > .99:
        print(tokens)
        text_data.append(tokens)

from gensim import corpora

dictionary = corpora.Dictionary(text_data)
corpus = [dictionary.doc2bow(text) for text in text_data]

import pickle

pickle.dump(corpus, open('corpus.pkl', 'wb'))
dictionary.save('dictionary.gensim')

import gensim

NUM_TOPICS = 10
NUM_WORDS = 15

ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=15)
ldamodel.save('model10.gensim')
topics = ldamodel.print_topics(num_words=NUM_WORDS)
for topic in topics:
    print(topic)

top_topics = ldamodel.top_topics(corpus)  # , num_words=15)

# Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
avg_topic_coherence = sum([t[1] for t in top_topics]) / NUM_TOPICS
print('Average topic coherence: %.4f.' % avg_topic_coherence)

from pprint import pprint

pprint(top_topics)

dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
corpus = pickle.load(open('corpus.pkl', 'rb'))

import pyLDAvis.gensim

lda10 = gensim.models.ldamodel.LdaModel.load('model10.gensim')
lda_display10 = pyLDAvis.gensim.prepare(lda10, corpus, dictionary, sort_topics=False)
pyLDAvis.display(lda_display10)
