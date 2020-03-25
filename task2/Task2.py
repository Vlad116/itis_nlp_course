import nltk
import pymorphy2
import pandas
from collections import Counter
import math
import numpy
morgh = pymorphy2.MorphAnalyzer()

# df1 = pandas.read_excel("C:/Users/vlada/PycharmProjects/KinopoiskPymorphy2/allmovies.xlsx", delimiter=',', encoding='UTF-8')
# print(df1)


def read_from_csv():
    film_reviews = pandas.read_csv('movies.csv', delimiter=',', encoding='windows-1251',
                                   names=['review', 'film_name', 'tone_of_the_review'])
    return film_reviews[['review']]

def to_normal_form(word):
    p = morgh.parse(word)[0]
    print(p.normal_form)
    return p.normal_form

#Все отзывы

f = open("../garbage/out.txt", 'w')

df = pandas.read_csv('movies.csv', delimiter=',', encoding='windows-1251',
                     names=['review', 'film_name', 'tone_of_the_review'])

for tuple in df.values:
    words = tuple[1].split()
    for word in words:
        f.write(to_normal_form(word) + '\n')

f.close()

#Все положительные отзывы

f = open("../garbage/good.txt", 'w')

df = pandas.read_csv('movies.csv', delimiter=',', encoding='windows-1251',
                     names=['review', 'film_name', 'tone_of_the_review'])

for tuple in df.values:
    if tuple[2] == 1:
        words = tuple[1].split()
        for word in words:
            f.write(to_normal_form(word) + '\n')

f.close()

#Все отрицательные отзывы

f = open("../garbage/good.txt", 'w')

df = pandas.read_csv('movies.csv', delimiter=',', encoding='windows-1251',
                     names=['review', 'film_name', 'tone_of_the_review'])

for tuple in df.values:
    if tuple[2] == -1:
        words = tuple[1].split()
        for word in words:
            f.write(to_normal_form(word) + '\n')

f.close()

# Все документы
positive = open("../garbage/positive.txt", "r")
negative = open("../garbage/negative.txt", "r")
all = open("../garbage/all.txt", "r")

# Списки слов по каждому документу
bagOfPositive = positive.read().split("\n")
bagOfNegative = negative.read().split("\n")
bagOfAll = all.read().split("\n")

#Убрать дублирующиеся слова
uniqueWords = set(bagOfPositive).union(set(bagOfNegative).union(set(bagOfAll)))

# Формируем для каждого документа словарь в котором ключ - это слово, а значение его частота в документе
mapOfWordsPositive = dict.fromkeys(uniqueWords, 0)
mapOfWordsNegative = dict.fromkeys(uniqueWords, 0)
mapOfAllWords = dict.fromkeys(uniqueWords, 0)

for word in bagOfPositive:
    mapOfWordsPositive[word] += 1
for word in bagOfNegative:
    mapOfWordsNegative[word] += 1
for word in bagOfAll:
    mapOfAllWords[word] += 1

# Фун-ия высчитывающая TF для каждого слова в документе
def computeTF(wordDictionary, bagOfWords):
    tfDictionary = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDictionary.items():
        tfDictionary[word] = count / float(bagOfWordsCount)
    return tfDictionary


# получили Map, где ключ - слово, а значение - его TF
tfPositive = computeTF(mapOfWordsPositive, bagOfPositive)
tfNegative = computeTF(mapOfWordsNegative, bagOfNegative)
tfAll = computeTF(mapOfAllWords, bagOfAll)

# функция высчитывающая значение IDF
def computeIDF(documents):
    N = len(documents)
    idfDictionary = dict.fromkeys(documents[0].keys(), 0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                idfDictionary[word] += 1

    for word, val in idfDictionary.items():
        idfDictionary[word] = math.log(N / float(val))
    return idfDictionary

# Получаем словарь, в котором ключ - слово, а значение - его IDF
idf = computeIDF([mapOfAllWords, mapOfWordsPositive, mapOfWordsNegative])

# фун-ия высчитывает tf-idf для каждого слова
def compute_TF_IDF(tfBagOfWords, idfs):
    tfIdf = {}
    for word, val in tfBagOfWords.items():
        tfIdf[word] = val * idfs[word]
    return tfIdf

# получаем словари для каждого документа, в котором ключ-слово, а значение tf-idf
tfIdfPositive = compute_TF_IDF(tfPositive, idf)
tfIdfNegative = compute_TF_IDF(tfNegative, idf)
tfIdfAll = compute_TF_IDF(tfAll, idf)

# Суммируем tf-idf каждого документа для каждого слова
df.loc['Total',:] = df.sum(axis=0)

# Ранжируем по метрике TF-IDF
total = sorted([(value, key) for (key, value) in df.tail(1).to_dict("index")["Total"].items()], reverse=True)
print(total)

# Берем топ 10 эл-тов
first10values = total[:10]
for item in first10values:
    print(item)


# Берем топ 10 среди положительных
positiveFrame = sorted([(value, key) for (key, value) in df.head(1).to_dict("index")[0].items()], reverse=True)

positiveFirst10Values = positiveFrame[:10]
for item in positiveFirst10Values:
    print(item)

# Берем топ 10 среди негативных

negativeFrame = sorted([(value, key) for (key, value) in df.head(2).to_dict("index")[1].items()], reverse=True)

negativeFirst10Values = negativeFrame[:10]
for item in negativeFirst10Values:
    print(item)