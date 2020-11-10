"""

Q4 - A Small Character Level LSTM
Train the LSTM for different sizes of input characters

"""
import re
import numpy as np
from pickle import dump
import tensorflow.keras as keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import pandas as pd
import matplotlib.pyplot as plt
import math

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
length = [5, 10, 20, 30, 40, 50]
epochs = 100
for k in range(len(length)):
    lines = list()
    for i in range(length[k], len(raw_text)):
        seq = raw_text[i-length[k]:i+1]
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
    history = model.fit(X, y, epochs=epochs, verbose=1)
    data = pd.DataFrame(history.history).values
    pd.DataFrame(history.history).to_csv('Q4-Train-Length-' + str(length[k]) + '.csv')
    
    # save the model and mapping to file
    model.save('model.h5')
    dump(mapping, open('mapping.pkl', 'wb'))

#plot the output    
epochs_lst = np.arange(epochs)
for k in range(len(length)):
    data = pd.read_csv('Q4-Train-Length-' + str(length[k]) + '.csv').values
    pp1 = [math.exp(x) for x in data[:, 1]]
    plt.plot(epochs_lst, pp1, label='k='+str(length[k]))
    plt.legend()
    
    
    
    
    
    
"""

Q4 (Testing) - A Small Character Level LSTM
Loads a trained LSTM and mapping and generates sentences

"""

from pickle import load
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

seed_text = 'you will probably be prepared to admit that you are not exceptional'
n_chars_to_predict = 100
seq_length = 20

# load the model and mapping
model = load_model('model.h5')
mapping = load(open('mapping.pkl', 'rb'))


# Make predictions
for k in range(n_chars_to_predict):
    # encode the characters as integers
    encoded = [mapping[char] for char in seed_text]
    # truncate sequences to a fixed length
    encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
    # one hot encode
    encoded = to_categorical(encoded, num_classes=len(mapping))
    # predict character
    yhat = model.predict_classes(encoded, verbose=0)
    
    # reverse map integer to character
    for char, index in mapping.items():
        if index == yhat:
            break
    seed_text += char

print("\n")
print(seed_text)