import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_prepare_lstm2(path, col="終値"):
    ##col=終わり値のデータの取得。日付けをインデックスに指定。欠損値を前方補完。加工後のデータ（終値）を出力。
    df=pd.read_csv(path)
    df["日付け"]=pd.to_datetime(df["日付け"])
    df.set_index("日付け",inplace=True)
    df.sort_index(inplace=True)

    df[col]=pd.to_numeric(df[col],errors="coerce") 
    df[col]=df[col].fillna(method="ffill")

    return df[[col]]

def scale_series(data):  
    ##正規化
    scaler=MinMaxScaler()
    data_scaled=scaler.fit_transform(data)
    return data_scaled, scaler

def data_seperate (data_scaled, window):
    ##トレーニングデータとテストデータを分ける
    training_data_len = int(np.ceil(len(data_scaled)*0.8))
    training_data = data_scaled[0:training_data_len:]
    testing_data = data_scaled[training_data_len-window: , :]
    return training_data_len, training_data, testing_data

def create_sequences(data, window=60):
    ##LSTMに渡すデータを出力
    x_train, y_train = [], []
    for i in range(window, len(data)):
        x_train.append(data[i-window:i,0]) ##入力
        y_train.append(data[i,0])          ##正解
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train_format = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1)) ##LSTMが受け付けるフォーマットに変換

    return x_train_format, y_train


def create_data(path, col='終値', window=60):
    # 前処理
    df = load_and_prepare_lstm2(path, col)
    scld, sclr = scale_series(df)
    training_data_len, train_d, tes_d = data_seperate(scld, window)
    
    # LSTM用データ作成
    x_train, y_train = create_sequences(train_d, window)
    x_test, y_test = create_sequences(tes_d, window)
    
    return df, sclr, x_train, y_train, x_test, y_test, training_data_len