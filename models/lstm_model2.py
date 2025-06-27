from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt 

def LSTM2_build(x_train):
    ##4段のLSTM層を導入し、また過学習を防ぐためその層ごとにDropout層を用意している。
    model = Sequential()
    model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1],1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mse')
    return model

def train_model(model,x_train,y_train,batch_size=32,epochs=100):
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
    return history

def predict_and_inverse(model,x_test,scaler):
    y_pred_scaled=model.predict(x_test)
    y_pred=scaler.inverse_transform(y_pred_scaled)
    return y_pred

def plot_predictions(data, training_data_len,y_pred):
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