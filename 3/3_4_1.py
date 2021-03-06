import numpy as np
from keras import models
from keras import layers
from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=1000)

print(train_data.shape)
print(train_labels.shape)

word_index = imdb.get_word_index()
reverse_word_index = {value:key for (key, value) in word_index.items()}
decoded_review = ' '.join(reverse_word_index.get(i - 3, '?') for i in train_data[0])

print(train_data[0])
print(decoded_review)
