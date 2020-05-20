import numpy as np
import keras
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Embedding, LSTM, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as ker

import pandas

batch_size = 128
epochs = 100
max_len = 15
# Импортировать данные
train_df = pandas.read_csv('my_reviews.csv')

df = pandas.read_csv("reviews.csv", encoding="utf-8")
reviews = df['text'].values
tokenizer = Tokenizer()

max_words = 10000  # Max size of the dictionary
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(reviews)
sequences = tokenizer.texts_to_sequences(reviews)
print('seq')
print(sequences)
print('reviews')
print(reviews)

# Flatten the list of lists resulting from the tokenization. This will reduce the list
# to one dimension, allowing us to apply the sliding window technique to predict the next word
text = [item for sublist in sequences for item in sublist]
print('text')
print(text)
vocab_size = len(tokenizer.word_index)
print(vocab_size)

# Training on 19 words to predict the 20th
sentence_len = 15
pred_len = 5
train_len = sentence_len - pred_len
seq = []

# Sliding window to generate train data
for i in range(len(text) - sentence_len):
    seq.append(text[i:i + sentence_len])
print(seq)

# Reverse dictionary to decode tokenized sequences back to words
reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
# print(reverse_word_map)

# Each row in seq is a 15 word long window. We append he first 10 words as the input to predict the next word
trainX = []
trainY = []

for i in seq:
    trainX.append(i[:train_len])
    trainY.append(i[-1])

# reverse_word_map[1514]

# define model
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout

model_2 = Sequential([
    Embedding(max_words, 50, input_length=train_len),
    LSTM(100, return_sequences=True),
    LSTM(100),
    Dense(100, activation='relu'),
    Dropout(0.1),
    Dense(max_words - 1, activation='softmax')
])

# Train model with checkpoints
model_2.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

filepath = "./model_2_weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
print(batch_size)

history = model_2.fit(np.asarray(trainX),
                      pandas.get_dummies(np.asarray(trainY)),
                      epochs=epochs,
                      batch_size=batch_size,
                      callbacks=callbacks_list,
                      verbose=1)


from keras.preprocessing.sequence import pad_sequences
import numpy as np

def gen(model, seq, max_len=max_len):
    print('GENERATE STRING BEGIN')
    ''' Generates a sequence given a string seq using specified model until the total sequence length
    reaches max_len'''

    # Tokenize the input string
    tokenized_sent = tokenizer.texts_to_sequences([seq])
    gen_res = seq.split(' ')
    # max_len = max_len + len(tokenized_sent[0])

    # If sentence is not as long as the desired sentence length, we need to 'pad sequence' so that
    # the array input shape is correct going into our LSTM. the `pad_sequences` function adds
    # zeroes to the left side of our sequence until it becomes 19 long, the number of input features.
    # while len(tokenized_sent[0]) < max_len:
    #     padded_sentence = pad_sequences(tokenized_sent[-19:], maxlen=19)
    #     op = model.predict(np.asarray(padded_sentence).reshape(1, -1))
    #     tokenized_sent[0].append(op.argmax() + 1)
    while len(gen_res) < max_len:
        padded_sentence = pad_sequences(tokenized_sent[-10:],maxlen=10)
        op = model.predict(np.asarray(padded_sentence).reshape(1,-1))
        tokenized_sent[0].append(op.argmax()+1)
        gen_res.append(reverse_word_map[op.argmax()+1])
    print('GENERATE STRING END')
    return " ".join(gen_res)
    # return " ".join(map(lambda x: reverse_word_map[x], tokenized_sent[0]))

sentences = [
    'Фильм явно не лишен смысла, всем советую посмотреть хоть раз',
    'Не представляю как можно было снять такой фильм не интересный',
    'Сценарий очень интересный, написан по книге не помню только какой',
    # 'Всё здорово, впечатление остаётся чудесное.  Но в какой-то момент задаёшь',
    # 'Фильм, после которого на душе хорошо. Душевно. Когда противоположности встречаются',
    # 'Самое сильное кино начала этого года, или конца прошлого, не',
    # 'До выхода этого фильма в Российский прокат осталось еще 20',
    # 'Саму атмосферу можно разделить на четыре стадии. Каждая из них',
    # 'В мультфильме, конечно, много исторических ошибок, но от этого он',
    # 'Довольно классический сюжет для мультфильма того времени: девушка пытается всеми',
    # 'окунаешься в детство и в ту атмосферу которую придает этот',
    # 'Он вызывает только приятные чувства и положительные эмоции как радость',
    # 'Если Вы знаете российскую историю начала ХХ века то, наверное',
    # 'Именно этот мультфильм пробудил во мне интерес к истории России',
    # 'А стоит смотреть так, как мы смотрим другие мультфильмы - мне',
    # 'Он красив, понравится зрительской аудитории, особенно детям. Добрый - а доброты',
    'Ну и пару слов о самом мультфильме, он очень неплох',
    'С множеством очень мощных и трогающих до глубины души моментов',
    'Рисовка очень качественная и красочная, лица изображены красиво и заманчиво',
    'С другой стороны здесь просто потрясающий русский дубляж нужно заметить',
]


for i, value in enumerate(sentences, 1):
    print(i, gen(model_2, value))

# print('Фильм неплохой, но можно было и лучше, а так неплохо')
# result = gen(model_2, 'Фильм неплохой, но можно было и лучше, а так неплохо')
# print(result)
#
# print('Муть какая-то')
# second_result = gen(model_2, 'Муть какая-то', 15)
# print(second_result)