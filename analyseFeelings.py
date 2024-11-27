import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt

# ler dataset
data = pd.read_pickle("merged_training.pkl") 

# filtrando o dataset para remover dados não rotulados
data = data[['content', 'sentiment']]
data = data.dropna()

X = data['content']
y = data['sentiment']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>') # tokenização das frases
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
pad_x = pad_sequences(sequences, maxlen=50, padding='post', truncating='post')

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(pad_x, y, test_size=0.2, random_state=42)

# Criar o modelo da rede neural
model = Sequential([
    Embedding(input_dim=5000, output_dim=64, input_length=50),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compilar o modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# treinando o modelo
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

# avaliando o modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Testando accuracy: {accuracy * 100:.2f}%")

# Visualizar o histórico de treinamento
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Fazer previsões
texts = ["i feel really happy today", "I'm really scared about the results.", "This is frustrating and annoying."]

sequences = tokenizer.texts_to_sequences(texts)
padded_texts = pad_sequences(sequences, maxlen=50, padding='post', truncating='post')
predictions = model.predict(padded_texts)
predicted_classes = [label_encoder.classes_[i] for i in predictions.argmax(axis=1)]
for text, emotion in zip(texts, predicted_classes):
    print(f"Texto: {text} -> Emoção prevista: {emotion}")
