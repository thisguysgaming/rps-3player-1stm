
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

st.title("Advanced 3-Player RPS Predictor with LSTM")

st.write("""
This app uses a Recurrent Neural Network (LSTM) to predict the next winner (Red, Green, or Blue)
in a 3-player Rock Paper Scissors game based on historical round outcomes.
""")

# Input winners
user_input = st.text_input("Enter winners separated by commas (e.g., Red, Green, Blue, ...):",
                           "Red, Green, Blue, Blue, Blue, Red, Green, Green, Green, Red")

if user_input:
    try:
        # Preprocess input
        winners = [w.strip().capitalize() for w in user_input.split(',')]
        le = LabelEncoder()
        encoded = le.fit_transform(winners)
        n_classes = len(le.classes_)

        if len(encoded) < 6:
            st.warning("Please enter at least 6 winners to train the model.")
        else:
            # Prepare data for LSTM
            sequence_length = 3
            X = []
            y = []
            for i in range(sequence_length, len(encoded)):
                X.append(encoded[i-sequence_length:i])
                y.append(encoded[i])
            X = np.array(X)
            y = to_categorical(y, num_classes=n_classes)
            X = X.reshape((X.shape[0], X.shape[1], 1))

            # Build LSTM model
            model = Sequential()
            model.add(LSTM(64, input_shape=(sequence_length, 1)))
            model.add(Dense(n_classes, activation='softmax'))
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model.fit(X, y, epochs=100, verbose=0)

            # Predict next winner
            last_seq = np.array(encoded[-sequence_length:]).reshape((1, sequence_length, 1))
            prediction = model.predict(last_seq)[0]
            predicted_index = np.argmax(prediction)
            predicted_winner = le.inverse_transform([predicted_index])[0]

            st.success(f"Predicted next winner: {predicted_winner}")
            st.write("Prediction probabilities:")
            for idx, prob in enumerate(prediction):
                st.write(f"{le.inverse_transform([idx])[0]}: {prob:.2%}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
