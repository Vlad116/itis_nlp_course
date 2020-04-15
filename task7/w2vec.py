import math
import string
from collections import Counter
from itertools import islice
import nltk
import numpy as np
import pandas
import pymorphy2
from keras import backend as ker

def get_text_in_normal_form(text):
    analyzer = pymorphy2.MorphAnalyzer()
    tokens = nltk.word_tokenize(text)
    normalized_words = []
    for token in tokens:
        if token in string.punctuation: continue
        normalized_words.append(analyzer.parse(token)[0].normal_form)
    return normalized_words


def get_notmalized_texts(data):
    normalized_texts = []
    i = 0
    for elem in data.values:
        print("Iteration: {}".format(i))
        normalized_texts.append(get_text_in_normal_form(elem[1]))
        i = i + 1
    return normalized_texts


def compute_tf_idf(corpus):
    def compute_tf(text):
        tf_text = Counter(text)
        for i in tf_text:
            tf_text[i] = tf_text[i] / float(len(text))
        return tf_text

    def compute_idf(word, corpus):
        return math.log10(len(corpus) / sum([1.0 for i in corpus if word in i]))

    documents_list = []
    for text in corpus:
        tf_idf_dictionary = {}
        computed_tf = compute_tf(text)
        for word in computed_tf:
            tf_idf_dictionary[word] = computed_tf[word] * compute_idf(word, corpus)
        documents_list.append(tf_idf_dictionary)
    return documents_list


def compute_frequency(all_text):
    frequency = {}
    for text in all_text:
        for word in text:
            count = frequency.get(word, 0)
            frequency[word] = count + 1
    return frequency


def get_tf_idf_vector(tfidf_list, frequency):
    vectors = []
    j = 0
    for text in tfidf_list:
        tfidf_vector = np.zeros(len(frequency))
        for w in list(text):
            for i, word in enumerate(list(frequency)):
                if word == w:
                    tfidf_vector[i] = text.get(w, 0)
        vectors.append(tfidf_vector)
        j += 1
        print("iteration: " + str(j))
    return vectors


def filter_by_reviews_title(data_frame, reviews_titles):
    return data_frame[~data_frame['title'].isin(reviews_titles)]


def recall_m(y_true, y_pred):
    true_positives = ker.sum(ker.round(ker.clip(y_true * y_pred, 0, 1)))
    possible_positives = ker.sum(ker.round(ker.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + ker.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = ker.sum(ker.round(ker.clip(y_true * y_pred, 0, 1)))
    predicted_positives = ker.sum(ker.round(ker.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + ker.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + ker.epsilon()))

# Cчитали все отзывы
filename = 'reviews.csv'
df = pandas.read_csv(filename, encoding="utf-8")
my_reviews = ['Матрица ', '1+1', 'Хоббит: Нежданное путешествие']

# Отделили тестовую выборку
test_data = df[df['title'].isin(my_reviews)]
df = filter_by_reviews_title(df, my_reviews)

print('--- Normalization data begin ---')
normalized_texts_train = get_notmalized_texts(df)
normalized_texts_predict = get_notmalized_texts(test_data)
print('--- Normalization end ---')

print('--- Computing TF-IDF begin ---')
tfidf_train = compute_tf_idf(normalized_texts_train)
tfidf_predict = compute_tf_idf(normalized_texts_predict)
print('--- Computing TF-IDF end ---')

# Выделяем 500 самых частых слов
frequency = compute_frequency(normalized_texts_train)
frequency = {k: v for k, v in sorted(frequency.items(), key=lambda item: item[1], reverse=True)}
frequency = dict(islice(frequency.items(), 0, 500))
