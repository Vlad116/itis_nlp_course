import nltk
import pymorphy2
import pandas
from collections import Counter
import math
import numpy

morgh = pymorphy2.MorphAnalyzer()

def read_from_txt():
    f = open('reviews.txt', encoding='utf-8')
    for line in f:
        words = nltk.word_tokenize(line)
        print(words)
        for word in words:
            print(word)

# nltk.word_tokenize(line))

# (?) def read_excel_file():
#       df = pandas.read_excel("reviews.xlsx")
#       df.columns = ["empty", "text", "rating", "name"]
#       df = df["text"]
#       for row in df:
#           print(row)

print('\n')

# film_reviews = pandas.read_csv('reviews.csv', delimiter=',', encoding='windows-1251',
#                                names=['review', 'film_name', 'tone_of_the_review'])

# morgh = pymorphy2.MorphAnalyzer()

out = open("out.txt", "w", encoding="UTF-8")

def read_from_csv():
    film_reviews = pandas.read_csv('../Task2/movies.csv', delimiter=',', encoding='windows-1251',
                                   names=['review', 'film_name', 'tone_of_the_review'])
    return film_reviews[['review']]

def to_normal_form(word):
    p = morgh.parse(word)[0]
    print(p.normal_form)
    return p.normal_form


# 1. Normalize all reviws

def normalize_all_reviws():

    film_reviews = read_from_csv()
    print(film_reviews + '\n')
    # data.loc[2:2, 'a':'b'] = 5, 6
    film_reviews_normalize_words = []

    for row in film_reviews.itertuples():
        words = nltk.word_tokenize(row.review)
        print(words)
        normalize_words = []

        # normalized_words = map(lambda word: to_normal_form(word), words)
        # print(normalized_words)
        for word in words:
            normal_form = to_normal_form(word)
            normalize_words.append(normal_form)
            print(normal_form)
            out.write(normal_form + '\n')
            # L = [1, 2, 3]
            # " ".join(str(x) for x in L)
            # '1 2 3'

        # film_reviews.loc[row.Index, 'review'] =
        film_reviews_normalize_words.append(normalize_words)
        # print(normalize_words)
        # print(row.review)
        # print(row.Index)

    return film_reviews_normalize_words

# (! ����������, � ��� ���-������)� ��� ����� �����, ����� ��� ���� ���-�� ��������������,
# ����� ����� ���� ��� ������� � ���� ����� ��������� ������ ����

# �������� ���� ����� ���������� ��������������� ������,
# �������� df ��� ����� �����, � ��� ����������� � csv

normalize_all_reviws()
out.close()



# 3. ��������� tf-idf ��� ������� ����� �� �����

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

# def tf_idf(words):
#     print('')

puncto = [',', '.', ':', '?', '�', '�', '-', '(', ')', '!', '\'', '�', ';', '�', '...']
words = []
# texts = f.read().replace('\n', ' ').split("SPLIT")
normalized_texts = normalize_all_reviws()
print(normalized_texts)

# print(len(texts))
# for text in normalized_texts:
#     tokens = nltk.word_tokenize(text)
#     normalized_words = []
#     for token in tokens:
#         if token in puncto: continue
#         word = analyzer.parse(token)[0]
#         normalized_words.append(word.normal_form)
#     normalized_texts.append(normalized_words)

tf_idf = compute_tf_idf(normalized_texts)



# 4. ��� ������� �����: c������ ��� ������� �� ���� ����������

def summ_all_metrics_for_words():
    print('')

# 5. ������������� ����� �� ������� tf-idf
def range_words_by_tf_idf():
    print('')

# all_unique_words = []
# for dictionary in tfidf:
#     for key in dictionary.keys():
#         all_unique_words.append(key)
# unique_words = numpy.unique(all_unique_words)
# print(len(numpy.unique(all_unique_words)))
# unique_dictionary = {}
# for word in unique_words:
#     unique_dictionary[word] = 0
#     for dictionary in tfidf:
#         unique_dictionary[word] += dictionary.get(word, 0)
# sorted_dictionary = {k: v for k, v in sorted(unique_dictionary.items(), key=lambda item: item[1], reverse=True)}
# print(sorted_dictionary, file=g)


# �������� ���������� �� ������� �������� ��������� ���� (��� tf-idf)
