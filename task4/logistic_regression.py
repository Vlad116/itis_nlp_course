import nltk
import numpy
import pandas
import pymorphy2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

def get_text_in_normal_form(text):
    analyzer = pymorphy2.MorphAnalyzer()
    punctuation = [',', '.', ':', '?', '«', '»', '-', '(', ')', '!', '\'', '—', ';', '”', '...', '#', '$', '%', '&']
    tokens = nltk.word_tokenize(text)
    normalized_words = []
    for token in tokens:
        if token in punctuation: continue
        normalized_words.append(analyzer.parse(token)[0].normal_form)
    return normalized_words


def get_vocabulary_and_bug_of_words(all_reviews_text, vocabulary):
    normalized_texts = []
    for elem in all_reviews_text:
        normalized_texts.append(get_text_in_normal_form(elem[1]))
    if len(vocabulary) == 0:
        vocabulary = numpy.unique(numpy.concatenate(normalized_texts))
    vectors = []
    n = 0
    for text in normalized_texts:
        bag_vector = numpy.zeros(len(vocabulary))
        for w in text:
            for i, word in enumerate(vocabulary):
                if word == w:
                    bag_vector[i] += 1
        vectors.append(bag_vector)
        n += 1
        print("Iteration: " + str(n))
    return vocabulary, vectors


def filter_data_frame_by_reviews_title(data_frame, reviews_titles):
    filtered_df = data_frame[~data_frame['title'].isin(reviews_titles)]
    return filtered_df

def print_stat(predicted, test_data):

    # Вычисляем Accuracy
    true_positives = 0
    k = 0
    for prediction in predicted:
        if prediction == test_data.values[k][2]:
            true_positives += 1
        k += 1

    print("Accuracy: {}".format(true_positives / len(predicted)))
    # Вычисляем метрики Precision, Recall, Fscore
    precision_recall_fscore = precision_recall_fscore_support(test_data['label'].values, predicted)
    print(f"Precision(-1, 0, 1) = {precision_recall_fscore[0]}")
    print(f"Recall(-1, 0, 1) = {precision_recall_fscore[1]}")
    print(f"Fscore(-1, 0, 1) = {precision_recall_fscore[2]}")

# Считали все отзовы
df = pandas.read_csv("reviews.csv", encoding="utf-8")
my_reviews = ['Матрица ', '1+1', 'Хоббит: Нежданное путешествие']

# Выделяем тестовую выборку
test_data = df[df['title'].isin(my_reviews)]
df = filter_data_frame_by_reviews_title(df, my_reviews)

# Получаем словарь всех слов и представление текстов в виде мешка слов
vocabulary, bag_of_words = get_vocabulary_and_bug_of_words(df.values, [])

# Подготавливаем мешок слов для тестовой выборки
vocabulary, test_bag_of_words = get_vocabulary_and_bug_of_words(test_data.values, vocabulary)

# Тренировка модели
print('--- TRAIN ---')
regression = LogisticRegression(max_iter=10000)
regression.fit(bag_of_words,df['label'])

# Применяем модель на тестовых текстах
print('--- PREDICT ---')
predicted = regression.predict(test_bag_of_words)
print_stat(predicted, test_data)

# # Создаем словари соотносящие все слова и коэффициенты для каждой тональности
# dict_of_negative = dict(zip(vocabulary, regression.coef_[0]))
# dict_of_neutral = dict(zip(vocabulary, regression.coef_[1]))
# dict_of_positive = dict(zip(vocabulary, regression.coef_[2]))
#
# # Сортитруем
# sorted_negative_dictionary = {k: v for k, v in sorted(dict_of_negative.items(), key=lambda item: item[1], reverse=True)}
# sorted_neutral_dictionary = {k: v for k, v in sorted(dict_of_neutral.items(), key=lambda item: item[1], reverse=True)}
# sorted_positive_dictionary = {k: v for k, v in sorted(dict_of_positive.items(), key=lambda item: item[1], reverse=True)}
#
# negative = open("negatives.txt", "w", encoding="utf-8")
# neutral = open("neutrals.txt", "w", encoding="utf-8")
# positive = open("positives.txt", "w", encoding="utf-8")
#
# print(sorted_negative_dictionary, file=negative)
# print(sorted_neutral_dictionary, file=neutral)
# print(sorted_positive_dictionary, file=positive)