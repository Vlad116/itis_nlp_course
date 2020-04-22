import nltk as nltk
import pandas
import pymorphy2
import numpy as np
import gensim.models
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support


def make_feature_vector(words, model, num_features):
    feature_vector = np.zeros((num_features,), dtype="float32")
    words_count = 0
    index2word_set = set(model.wv.index2word)
    for word in words:
        if word in index2word_set:
            words_count = words_count + 1
            feature_vector = np.add(feature_vector, model[word])

    feature_vector = np.divide(feature_vector, words_count)
    return feature_vector


def get_avg_feature_vectors(reviews, model, num_features):
    counter = 0
    review_feature_vectors = np.zeros((len(reviews), num_features), dtype='float32')
    for review in reviews:
        review_feature_vectors[counter] = make_feature_vector(review, model, num_features)
        counter = counter + 1
    return review_feature_vectors


def get_text_in_normal_form(text):
    analyzer = pymorphy2.MorphAnalyzer()
    normalized_text = []
    tokens = nltk.word_tokenize(text)
    for token in tokens:
        normalized_text.append(analyzer.parse(token)[0].normal_form)
    return normalized_text


def filter_by_reviews_title(data_frame, reviews_titles):
    return data_frame[~data_frame['title'].isin(reviews_titles)]


# Считываем все отзывы
df = pandas.read_csv("reviews.csv", encoding="utf-8")
df['text'] = df['text'].map(lambda x: get_text_in_normal_form(x))
my_reviews = ['Матрица ', '1+1', 'Хоббит: Нежданное путешествие']

# Отделяем тестовые отзывы
test_data = df[df['title'].isin(my_reviews)]
df = filter_by_reviews_title(df, my_reviews)

# Создаем модель word2vec
model = gensim.models.Word2Vec(sentences=df['text'], min_count=3, iter=300)

# Синонимы
print(model.wv.most_similar(positive=['фильм'], topn=5))
print(model.wv.most_similar(positive=['плохой'], topn=5))
print(model.wv.most_similar(positive=['работа'], topn=5))
print(model.wv.most_similar(positive=['оценка'], topn=5))
print(model.wv.most_similar(positive=['шок'], topn=5))

# Получаем вектора
trainDataVectors = get_avg_feature_vectors(df['text'], model, 100)
testDataVectors = get_avg_feature_vectors(test_data['text'], model, 100)

# Тренируем модель (лучший результат был у RF, поэтому выбираем его)
rfc = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=0)
rfc.fit(trainDataVectors, df['label'])

# Применяем модель к текстовым текстам
predicted = rfc.predict(testDataVectors)

# Вычисляем Accuracy
true_positives = 0
k = 0

for prediction in predicted:
    if prediction == test_data.values[k][2]:
        true_positives += 1
    k += 1

print("Accuracy: {}".format(true_positives / len(predicted)))

# Вычисляем Precision, Recall, Fscore
precision_recall_fscore = precision_recall_fscore_support(test_data['label'].values, predicted)
print(f"Precision(-1, 0, 1) = {precision_recall_fscore[0]}")
print(f"Recall(-1, 0, 1) = {precision_recall_fscore[1]}")
print(f"Fscore(-1, 0, 1) = {precision_recall_fscore[2]}")