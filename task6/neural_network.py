import keras
import numpy as np
from keras import backend as ker
from collections import Counter
import math
from itertools import islice
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import nltk
import pandas
import pymorphy2


# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score
# from sklearn.metrics import f1_score
# from sklearn.metrics import accuracy_score
# Using TensorFlow backend.

def get_text_in_normal_form(text):
    analyzer = pymorphy2.MorphAnalyzer()
    punctuation = [',', '.', ':', '?', '«', '»', '-', '(', ')', '!', '\'', '—', ';', '”', '...', '#', '$', '%', '&']
    tokens = nltk.word_tokenize(text)
    normalized_words = []
    for token in tokens:
        if token in punctuation: continue
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
tf_idf_train = compute_tf_idf(normalized_texts_train)
tf_idf_predict = compute_tf_idf(normalized_texts_predict)
print('--- Computing TF-IDF end ---')

# Выделяем 500 самых частых слов
frequency = compute_frequency(normalized_texts_train)
frequency = {k: v for k, v in sorted(frequency.items(), key=lambda item: item[1], reverse=True)}
frequency = dict(islice(frequency.items(), 0, 500))

print("--- Getting vectors ---")
# Получаем вектора для тренировки нейронной сети и для предсказания
x_train_vector = get_tf_idf_vector(tf_idf_train, frequency)
y_train_vector = get_tf_idf_vector(tf_idf_predict, frequency)
print("--- End getting vectors ---")

# # tf-idf
# tfidf = TfidfVectorizer(max_features=500)
# tfidf.fit_transform(reviews_training)
# x_train_tfidf = tfidf.fit_transform(reviews_training)
# x_test_tfidf = tfidf.fit_transform(reviews_testing)
#
# # building model
# y_train_categorical = keras.utils.to_categorical(all_reviews["label"], 3)
# y_test_categorical = keras.utils.to_categorical(y_test, 3)


num_classes = 3
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)

# Настройка слоев нейронной сети
model = Sequential()
# Input - Layer
model.add(Dense(512, input_shape=(500,)))
model.add(Activation('relu'))
# Cкрытый полносвязный
model.add(Dense(512))
# Dropout слой
model.add(Dropout(0.5))
# model.add(Dense(num_classes))
# model.add(Activation('softmax'))
# model.add(Dense(512))
model.add(Dense(3, activation='softmax'))
# model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', f1_m, precision_m, recall_m])

print("---TRAINING---")
batch_size = 32
epochs = 10
model.fit(np.array(x_train_vector), np.array(df[['label']]), epochs=epochs, batch_size=batch_size)
# np.array(df[['label']])

print("---PREDICT---")
loss, accuracy, f1_score, precision, recall = model.evaluate(y_train_vector, test_data['label'], verbose=0)

print("Test loss:", loss)
print("Test accuracy:", accuracy)
print("F1:", f1_score)
print("Precision:", precision)
print("Recall:", recall)
