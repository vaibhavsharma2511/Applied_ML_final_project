# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the dataset
data = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

# Preprocess the data
data['Review'] = data['Review'].apply(lambda x: x.lower())

# Tokenization
max_features = 5000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(data['Review'].values)
X = tokenizer.texts_to_sequences(data['Review'].values)
X = pad_sequences(X)

# Label Encoding
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(data['Liked'])
Y = to_categorical(Y, num_classes=2)

# Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(Embedding(max_features, 128, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
batch_size = 32
model.fit(X_train, Y_train, epochs=5, batch_size=batch_size, validation_data=(X_test, Y_test), verbose=2)

# Save the model and tokenizer
model.save('sentiment_model.h5')
pickle.dump(tokenizer, open('tokenizer.pkl', 'wb'))
