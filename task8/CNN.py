import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input, Embedding, Conv1D, GlobalMaxPool1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as ker
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from keras.objectives import categorical_crossentropy
import pandas

max_words = 500
batch_size = 32
epochs = 10
max_len = 40

def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix

# def recall_m(y_true, y_pred):
#     true_positives = ker.sum(ker.round(ker.clip(y_true * y_pred, 0, 1)))
#     possible_positives = ker.sum(ker.round(ker.clip(y_true, 0, 1)))
#     recall = true_positives / (possible_positives + ker.epsilon())
#     return recall
#
# def precision_m(y_true, y_pred):
#     true_positives = ker.sum(ker.round(ker.clip(y_true * y_pred, 0, 1)))
#     predicted_positives = ker.sum(ker.round(ker.clip(y_pred, 0, 1)))
#     precision = true_positives / (predicted_positives + ker.epsilon())
#     return precision
#
# def f1_m(y_true, y_pred):
#     precision = precision_m(y_true, y_pred)
#     recall = recall_m(y_true, y_pred)
#     return 2 * ((precision * recall) / (precision + recall + ker.epsilon()))

df_train = pandas.read_csv("reviews.csv")
df_test = pandas.read_csv("my_reviews.csv")
df_val = pandas.read_csv("val.csv")

df_train.head()

print("Preparing the Tokenizer...")
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df_train["text"])

print('Vectorizing sequence data...')
x_train = tokenizer.texts_to_sequences(df_train["text"])
x_test = tokenizer.texts_to_sequences(df_test["text"])
x_val = tokenizer.texts_to_sequences(df_val["text"])

x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)
x_val = pad_sequences(x_val, maxlen=max_len)

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

# x_train

print('Convert class vector to binary class matrix '
      '(for use with categorical_crossentropy)')
num_classes = 3
y_train = keras.utils.to_categorical(df_train["label"], num_classes)
y_val = keras.utils.to_categorical(df_val["label"], num_classes)
print('y_train shape:', y_train.shape)
print('y_val shape:', y_val.shape)

# y_val

print('Building model sequentially')
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=128, input_length=max_len))
model.add(Conv1D(128, 3))
model.add(Activation("relu"))
model.add(GlobalMaxPool1D())
model.add(Dense(num_classes))
model.add(Activation('softmax'))

print(model.layers)

print(model.to_yaml())

# from keras.utils import plot_model
# plot_model(model, to_file='model.png', show_shapes=True)

# from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot
# SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

from keras.objectives import categorical_crossentropy
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy',f1_score, precision_score, recall_score])

# from keras.callbacks import TensorBoard
# tensorboard=TensorBoard(log_dir='./logs', write_graph=True, write_images=True)
# from keras.callbacks import EarlyStopping
# early_stopping=EarlyStopping(monitor='val_loss')


history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1,
                    # callbacks=[tensorboard, early_stopping]
                    )

loss, accuracy, f1_score, precision, recall = model.evaluate(x_val, y_val, batch_size=batch_size, verbose=1)

print('\n')
print("Test loss:", loss)
print("Test accuracy:", accuracy)
print("F1:", f1_score)
print("Precision:", precision)
print("Recall:", recall)

results = model.predict(x_test, batch_size=batch_size, verbose=1)

from gensim.models import KeyedVectors

vocab_size = len(tokenizer.word_index) + 1

# # load embeddings
# w2v = KeyedVectors.load("word2vec.model")
#
# # create a weight matrix for words in training docs
# embedding_matrix = np.random.random((vocab_size, 100))
# for word, i in tokenizer.word_index.items():
#     if word in w2v.wv:
#         embedding_matrix[i] = w2v.wv[word]

embedding_dim = 300
embedding_matrix = create_embedding_matrix(
     '../../model.txt',
    tokenizer.word_index, embedding_dim)

model = Sequential()
model.add(Embedding(input_dim=vocab_size, weights=[embedding_matrix],
                    output_dim=embedding_dim, input_length=max_len,
                   trainable=True))
model.add(Conv1D(128, 3))
model.add(Activation("relu"))
model.add(GlobalMaxPool1D())
model.add(Dense(num_classes))
model.add(Activation('softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy', f1_score, precision_score, recall_score])

# from keras.callbacks import TensorBoard
# tensorboard=TensorBoard(log_dir='./logs', write_graph=True, write_images=True)
# from keras.callbacks import EarlyStopping
# early_stopping=EarlyStopping(monitor='val_loss')

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1,
                    # callbacks=[tensorboard, early_stopping]
                    )

loss, accuracy, f1_score, precision, recall = model.evaluate(x_val, y_val, batch_size=batch_size, verbose=1)
print('\n')
print("Test loss:", loss)
print("Test accuracy:", accuracy)
print("F1:", f1_score)
print("Precision:", precision)
print("Recall:", recall)