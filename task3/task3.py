import pymorphy2
import pandas
morgh = pymorphy2.MorphAnalyzer()


def read_from_csv():
    film_reviews = pandas.read_csv('films.csv', delimiter=',', encoding='utf-8')
    return film_reviews[['review']]


def to_normal_form(word):
    p = morgh.parse(word)[0]
    print(p.normal_form)
    return p.normal_form


f = open("task3.txt", 'w')

# считали из файла
df = pandas.read_csv('films.csv', delimiter=',', encoding = "UTF-8")

valid = df["title"].isin(['Матрица', '1+1', 'Хоббит: Нежданное путешествие'])
test = df[valid]
del test['title']

# удаляем наши данные (оставляем только для обучения)

df = df.loc[df['title'] != "Матрица"]
df = df.loc[df['title'] != "1+1"]
df = df.loc[df['title'] != "Хоббит: Нежданное путешествие"]
del df['title']

# f = open('reviews.txt', encoding='utf-8')
# out = open("out.txt", "w", encoding="UTF-8")

positive = open('AllPositive.txt', "w", encoding="UTF-8", errors='ignore')
negative = open('AllNegative.txt', "w", encoding="UTF-8", errors='ignore')
neutral = open('AllNeutral.txt', "w", encoding="UTF-8", errors='ignore')

index = 0
for tuple in df.values:
    words = tuple[0].split()
    if tuple[1] == '0':
        for word in words:
            wordNF = morgh.parse(word)[0]
            neutral.write(wordNF.normal_form + '\n')
    elif tuple[1] == '1':
        for word in words:
            wordNF = morgh.parse(word)[0]
            positive.write(wordNF.normal_form + '\n')
    elif tuple[1] == "-1":
        for word in words:
            wordNF = morgh.parse(word)[0]
            negative.write(wordNF.normal_form + '\n')
    print(index)
    index += 1  # смотреть на какой строке

# Read words
positive = open('AllPosititve.txt', "r", encoding='utf-8', errors='ignore')
negative = open('AllNegative.txt', "r", encoding='utf-8', errors='ignore')
neutral = open('AllNeutral.txt', "r", encoding='utf-8', errors='ignore')

# Get words list for categories
bagOfPositive = positive.read().split('\n')
bagOfNegative = negative.read().split('\n')
bagOfNeutral = neutral.read().split('\n')

# filter words
uniqueWords = len(set(bagOfPositive).union(set(bagOfNegative).union(set(bagOfNeutral))))


# функция принимающая на вход отзыв и выдает его принадлежность к каждому из трех классов (позитивный, негативный и нейтральность)
def getPredictionFor(words):

    allCount = len(bagOfNegative) + len(bagOfNeutral) + len(bagOfPositive)
    positivePrediction = 0
    positivePredictions = []
    neutralPrediction = 0
    neutralPredictions = []
    negativePrediction = 0
    negativePredictions = []

    # приводим слова в нормальную форму и раскидываем их значения по соответствующим массивам
    for word in words:
        if bagOfNeutral.count(word) + bagOfNegative.count(word) + bagOfPositive.count(word) != 0:
            wordNF = morgh.parse(word)[0]
            word = wordNF.normal_form

            countWordPositive = bagOfPositive.count(word)
            positiveMarkWord = (countWordPositive + 1) / (len(bagOfPositive) + uniqueWords)
            positivePredictions.append(positiveMarkWord)

            countWordNeutral = bagOfNeutral.count(word)
            neutralMarkWord = (countWordNeutral + 1) / (len(bagOfNeutral) + uniqueWords)
            neutralPredictions.append(neutralMarkWord)

            countWordNegative = bagOfNegative.count(word)
            negativeMarkWord = (countWordNegative + 1) / (len(bagOfNegative) + uniqueWords)
            negativePredictions.append(negativeMarkWord)

    lastPositivePrediction = 0
    lastNegativePrediction = 0
    lastNeutralPrediction = 0

    # Умножаем значения в каждом массиве, чтобы получить результирующее значение о вероятности принадлежности к классу
    positivePrediction = len(bagOfPositive) / allCount
    for mark in positivePredictions:
        while positivePredictions > 0:
            lastPositivePrediction = positivePrediction
            positivePrediction = positivePrediction * mark

    negativePrediction = len(bagOfNegative) / allCount
    for mark in negativePredictions:
        while negativePredictions > 0:
            lastNegativePrediction = negativePrediction
            negativePrediction = negativePrediction * mark

    neutralPrediction = len(bagOfNeutral) / allCount
    for mark in neutralPredictions:
        while neutralPredictions > 0:
            lastNeutralPrediction = neutralPrediction
            neutralPrediction = neutralPrediction * mark

    return lastPositivePrediction, lastNegativePrediction, lastNeutralPrediction


# получение результатов
index = 0
correct = 0

for tuple in test.values:
    words = tuple[0].split()
    result = getPredictionFor(words)
    predictedValue = 2
    if (result[0] > result[1]) & (result[0] > result[2]):
        predictedValue = 1
    elif (result[1] > result[0]) & (result[1] > result[2]):
        predictedValue = 0
    elif (result[2] > result[1]) & (result[2] > result[0]):
        predictedValue = -1
    print(index, tuple[1], predictedValue)
    if tuple[1] == f"{predictedValue}":
        correct += 1
    index += 1
print(f"correct {correct} accuracy {correct / test.shape[0]}")

neutral.close()
negative.close()
positive.close()
