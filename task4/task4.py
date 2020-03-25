import morph
import pymorphy2
import pandas
from sklearn.linear_model import LogisticRegression

morgh = pymorphy2.MorphAnalyzer()

#считали из файла
df = pandas.read_csv("task4/films.csv", encoding="utf-8")

# текстовые данные
valid = df["title"].isin(['Матрица', "1+1", "Хоббит: Нежданное путешествие"])
test = df[valid]
del test['title']

# удалили наши данные(оставили данные для обучения)
df = df.loc[df['title'] != "Матрица"]
df = df.loc[df['title'] != "1+1"]
df = df.loc[df['title'] != "Хоббит: Нежданное путешествие"]
del df['title']

# f = open("task4.txt",'w')
tupleIndex = 0
allWords = open("AllWords.txt", "w")

for tuple in df.values:
    tupleIndex+=1
    print(tupleIndex)
    words = tuple[0].split()
    for word in words:
        wordNF = morph.parse(word)[0]
        allWords.write(wordNF.normal_form+"\n")

for tuple in test.values:
    tupleIndex+=1
    print(tupleIndex)
    words = tuple[0].split()
    for word in words:
        wordNF = morph.parse(word)[0]
        allWords.write(wordNF.normal_form+"\n")

allWords.close()
allWordsRead = open("AllWords.txt", "r")
allWordsList = allWordsRead.read().split("\n")
uniqueWords = set(allWordsList)
allWordsRead.close()

def getBagOfWords(df, uniqWords):
    index = 0
    resDF = pandas.DataFrame(columns=uniqWords)
    answerDF = pandas.DataFrame(columns=["Answer"])
    for tuple in df.values:
        bagDictionary = dict.fromkeys(uniqWords, 0)
        index += 1
        print(index)
        words = tuple[0].split()
        for word in words:
            wordNF = morph.parse(word)[0].normal_form
            bagDictionary[wordNF] += 1
        resDF.loc[index] = list(bagDictionary.values())
        answerDF.loc[index] = tuple[1]
    return resDF, answerDF


studyDFs = getBagOfWords(df, uniqueWords)
testDFs = getBagOfWords(test, uniqueWords)

print(studyDFs[0].shape, studyDFs[1].shape)

X = studyDFs[0].to_numpy()
y = studyDFs[1]["Answer"]

clf = LogisticRegression(random_state=0).fit(X, y)
y_prediction = clf.predict(testDFs[0])

A = 0
B = 0
C = 0
D = 0
E = 0
F = 0
G = 0
H = 0
I = 0

for index in range(len(y_prediction)):

    if (testDFs[1]["Answer"].values[index] == "1") & (y_prediction[index] == "1"):
        A += 1
    elif (testDFs[1]["Answer"].values[index] == "1") & (y_prediction[index] == "0"):
        B += 1
    elif (testDFs[1]["Answer"].values[index] == "1") & (y_prediction[index] == "-1"):
        C += 1
    elif (testDFs[1]["Answer"].values[index] == "0") & (y_prediction[index] == "1"):
        D += 1
    elif (testDFs[1]["Answer"].values[index] == "0") & (y_prediction[index] == "0"):
        E += 1
    elif (testDFs[1]["Answer"].values[index] == "0") & (y_prediction[index] == "-1"):
        F += 1
    elif (testDFs[1]["Answer"].values[index] == "-1") & (y_prediction[index] == "1"):
        G += 1
    elif (testDFs[1]["Answer"].values[index] == "-1") & (y_prediction[index] == "0"):
        H += 1
    elif (testDFs[1]["Answer"].values[index] == "-1") & (y_prediction[index] == "-1"):
        I += 1

precision_class1 = A/(A+D+G)
precision_class2 = E/(B+E+H)
precision_class3 = I/(C+F+I)

recall_class1 = A/(A+B+C)
recall_class2 = E/(E+D+F)
recall_class3 = I/(I+G+H)

F_mera_class1 = (precision_class1*2*recall_class1)/(precision_class1+recall_class1)
F_mera_class2 = (precision_class2*2*recall_class2)/(precision_class2+recall_class2)
F_mera_class3 = (precision_class3*2*recall_class3)/(precision_class3+recall_class3)

precision = (precision_class1+precision_class2+precision_class3)/3
recall = (recall_class1+recall_class2+recall_class3)/3
f_mera = (F_mera_class1+F_mera_class2+F_mera_class3)/3

accuracy = (A + E + I) / (A + B + C + D + E + F + G + H + I)

print(f"Precision = {precision}")
print(f"Recall = {recall}")
print(f"F mera = {f_mera}")
print(f"Accuracy = {accuracy}")

dictionary_class1 = dict.fromkeys(uniqueWords, 0)
dictionary_class2 = dict.fromkeys(uniqueWords, 0)
dictionary_class3 = dict.fromkeys(uniqueWords, 0)

for index in range(len(clf.coef_[0])):
    print(index)
    dictionary_class1[list(dictionary_class1.keys())[index]] = clf.coef_[0][index]
    dictionary_class2[list(dictionary_class2.keys())[index]] = clf.coef_[1][index]
    dictionary_class3[list(dictionary_class3.keys())[index]] = clf.coef_[2][index]

finalDictionary1 = sorted((value, key) for (key, value) in dictionary_class1.items())
finalDictionary2 = sorted((value, key) for (key, value) in dictionary_class2.items())
finalDictionary3 = sorted((value, key) for (key, value) in dictionary_class3.items())

def getTop10Word(list):
    for index in range(10):
        print(list[-index])
    for index in range(10):
        print(list[index])

getTop10Word(finalDictionary1)
getTop10Word(finalDictionary2)
getTop10Word(finalDictionary3)