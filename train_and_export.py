# train_and_export.py
import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from keras.preprocessing import sequence
import keras2onnx
import onnx

top_words = 5000
max_words = 500
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

model = Sequential()
model.add(Embedding(top_words, 32, input_length=max_words))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=2, batch_size=128, validation_data=(X_test, y_test))

onnx_model = keras2onnx.convert_keras(model, model.name)
onnx.save_model(onnx_model, "imdb_sentiment.onnx")
print("✅ Model saved to imdb_sentiment.onnx")
