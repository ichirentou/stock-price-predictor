

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

def build_lstm_model(x_train):
    ##モデルの構築
    model=Sequential()
    model.add(LSTM(50, activation='relu',input_shape=(x_train.shape[1],1)))
    model.add(Dense(1))
    model.compile(optimizer='adam',loss='mse')
    return model

def train_model(model,X_train,y_train,epochs=100,batch_size=32):
    ##モデルのトレーニング
    early_stop=EarlyStopping(monitor='loss',patience=10,restore_best_weights=True)
    history = model.fit(X_train,y_train, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stop])
    return history

def predict_and_inverse(model, X_test, scaler):
    ##モデルの予測と正規化の逆変換
    y_pred_scaled=model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    return y_pred

def plot_predictions(data, training_data_len,y_pred):
    ##グラフにプロット
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = y_pred

    plt.figure(figsize=(16,6))
    plt.title('Model')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('終値', fontsize=14)
    plt.plot(train['終値'])
    plt.plot(valid[['終値', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()