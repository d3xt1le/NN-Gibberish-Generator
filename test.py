import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import RMSprop
import random, nltk, pronouncing
from nltk.corpus import words

# Download the word corpus from NLTK
nltk.download('words')
english_words = words.words()

# create phonetic representations of words
phonetic_reps = [pronouncing.phones_for_word(word) for word in english_words]

# create character list of words
tokenized_words = [[char for char in word] for word in english_words]

# create set of vocab characters
vocab_chars = set(''.join(english_words))

# Create a mapping of unique characters to integers
char_to_index = { char : idx for idx, char in enumerate(vocab_chars)}
index_to_char = { idx : char for char, idx in char_to_index.items()}

# define length of sequence
sequence_length = 10

# create sequences and next character in sequence
sequences = []
next_chars = []

for word in tokenized_words:
    for idx in range(len(word) - sequence_length):
        sequences.append(word[idx : idx + sequence_length])
        next_chars.append(word[idx + sequence_length])

# vectorize sequences
X = np.zeros((len(sequences), sequence_length, len(vocab_chars)), dtype=np.bool_)
y = np.zeros((len(sequences), len(vocab_chars)), dtype=np.bool_)

for i, sequence in enumerate(sequences):
    for t, char in enumerate(sequence):
        X[i, t, char_to_index[char]] = 1
    y[i, char_to_index[next_chars[i]]] = 1

# define LSTM Model
model = Sequential()
model.add(LSTM(128, input_shape=(sequence_length, len(vocab_chars))))
model.add(Dense(len(vocab_chars), activation='softmax'))
optimizer = RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# train model
model.fit(X, y, batch_size=100, epochs=10)


# The temperature parameter in the sample function controls the randomness of the predictions. 
# A higher temperature will result in more random predictions, while a lower temperature will result in less random predictions. 
# You can adjust this parameter to control the “creativity” of the model.
def sample(preds, temperature=1.3):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_word(length):
    # Choose a random sequence from your training data
    start_index = random.randint(0, len(sequences) - 1)
    sentence = sequences[start_index]
    generated = ''.join(sentence)

    for _ in range(length):
        x_pred = np.zeros((1, sequence_length, len(vocab_chars)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_to_index[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds)
        next_char = index_to_char[next_index]

        generated += next_char
        sentence = sentence[1:] + [next_char]
    return generated

for _ in range(10):
    print(f"Word: {generate_word(3)}")
