"""

Q4 - A Small Character Level LSTM
Train the LSTM for different sizes of input characters

"""
import re
import numpy as np
from pickle import dump
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Load and clean a text file
def fClean_Load(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    # Clean text
    words = re.findall(r'[a-z\.]+', text.lower())
    return ' '.join(words)

# load text
raw_text = fClean_Load('Unhappy.txt')


# organize into sequences of characters
#################################################################################
############################# Select a Size of 5/10/20/30/40/50 ####################
##################################################################################
length = 
lines = list()
for i in range(length, len(raw_text)):
    seq = raw_text[i-length:i+1]
    lines.append(seq)
print('Total lines: %d' % len(lines))

print(lines)


## Character mapping

chars = sorted(list(set(''.join(lines))))
mapping = dict((c, i) for i, c in enumerate(chars))

sequences = list()
for line in lines:
	encoded_seq = [mapping[char] for char in line]
	sequences.append(encoded_seq)
    
vocab_size = len(mapping)
print('Vocabulary Size: %d' % vocab_size)


## Input for the Model
sequences = np.array(sequences)
X1, y = sequences[:,:-1], sequences[:,-1]
temp = [to_categorical(x, num_classes=vocab_size) for x in X1]
X = np.array(temp)
y = to_categorical(y, num_classes=vocab_size)

# LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())

# Model Compiling and Fiting
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=100, verbose=1)


# save the model and mapping to file
model.save('model.h5')
dump(mapping, open('mapping.pkl', 'wb'))










