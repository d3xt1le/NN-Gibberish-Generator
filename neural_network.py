import nltk
import pronouncing
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from nltk.corpus import words
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# download dataset
nltk.download('words')
english_words = words.words()

# convert english words to phonetic representations
phonetic_representations = [pronouncing.phones_for_word(word) for word in english_words]

# tokenize words into characters
tokenized_words = [[char for char in word] for word in english_words]

# create character vocab
vocab = set(''. join(english_words))

# create a dictionary mapping characters to numerical indices
char_to_index = {char : idx for idx, char in enumerate(vocab)}
index_to_char = {idx : char for char, idx in char_to_index.items()}

def one_hot_encode(word, vocab_size):
    encoding = np.zeros((len(word), vocab_size), dtype=np.float32)
    for idx, char in enumerate(word):
        encoding[idx, char_to_index[char]] = 1.0
    return encoding

word = "example"
encoded_word = one_hot_encode(word, len(vocab))

# Print some information
print(f"\nTotal Number of English words: {len(english_words)}\n" )

# Define the LSTM model
def build_lstm_model(vocab_size, embedding_dim, rnn_units):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=None),
        tf.keras.layers.LSTM(rnn_units, return_sequences=True),
        tf.keras.layers.Dense(vocab_size, activation='softmax')
    ])
    return model

# Parameters
vocab_size = len(vocab)  # Vocabulary size
embedding_dim = 256      # Embedding dimension
rnn_units = 512          # Number of LSTM units

# Build the LSTM model
lstm_model = build_lstm_model(vocab_size, embedding_dim, rnn_units)

# Compile the model
lstm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy')

# Print model summary
lstm_model.summary()

# Split dataset into training and validation sets
# Convert words to one-hot encoded representations
labels = phonetic_representations
encoded_data = [one_hot_encode(word, len(vocab)) for word in tokenized_words]
X_train, X_val, y_train, y_val = train_test_split(encoded_data, labels, test_size=0.2, random_state=42)

# Train the model
history = lstm_model.fit(X_train, y_train, batch_size=64, epochs=1, validation_data=(X_val, y_val))

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

