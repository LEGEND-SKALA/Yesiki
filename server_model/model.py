# model.py
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
from keras.utils import plot_model
import os
import math
from sklearn.metrics import mean_squared_error
from server_model.config import DATA_PATH, MODEL_SAVE_PATH, MODEL_PLOT_PATH, MODEL_SHAPES_PLOT_PATH, PREDICTION_PLOT_PATH

def train_model(dataset):
    training_set = dataset.iloc[:-7][['광어 소비량(g)', '연어 소비량(g)', '장어 소비량(g)']].values
    sc = MinMaxScaler()
    training_set_scaled = sc.fit_transform(training_set)

    X_train, y_train = [], []
    for i in range(30, len(training_set_scaled)):
        X_train.append(training_set_scaled[i-30:i])
        y_train.append(training_set_scaled[i])

    X_train, y_train = np.array(X_train), np.array(y_train)

    model = Sequential([
        LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(units=64),
        Dropout(0.2),
        Dense(units=3)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=30, batch_size=8, verbose=0)
    model.save(MODEL_SAVE_PATH)
    plot_model(model, to_file=MODEL_PLOT_PATH)
    plot_model(model, to_file=MODEL_SHAPES_PLOT_PATH, show_shapes=True)
    return model, sc

def process(dataset):
    target_columns = ['광어 소비량(g)', '연어 소비량(g)', '장어 소비량(g)']
    raw_values = dataset[target_columns].values

    sc = MinMaxScaler()
    scaled = sc.fit_transform(raw_values)

    X_test, y_test = [], []
    for i in range(len(scaled) - 30 - 7 + 1):
        X_test.append(scaled[i:i+30])
        y_test.append(scaled[i+30:i+30+7])

    X_test, y_test = np.array(X_test), np.array(y_test)
    model = load_model(MODEL_SAVE_PATH)
    preds = model.predict(X_test)
    preds_inv = sc.inverse_transform(preds)
    y_test_inv = sc.inverse_transform(y_test[:, -1])

    rmse = math.sqrt(mean_squared_error(y_test_inv, preds_inv))
    print(f"📊 RMSE: {rmse}")

    if rmse >= 5:
        print("🔁 RMSE 기준 초과로 재학습 수행 중...")
        model, sc = train_model(dataset)
        preds = model.predict(X_test)
        preds_inv = sc.inverse_transform(preds)

    # 시각화
    plt.clf()
    plt.plot(y_test_inv[:, 0], label='광어 실제')
    plt.plot(preds_inv[:, 0], label='광어 예측')
    plt.plot(y_test_inv[:, 1], label='연어 실제')
    plt.plot(preds_inv[:, 1], label='연어 예측')
    plt.plot(y_test_inv[:, 2], label='장어 실제')
    plt.plot(preds_inv[:, 2], label='장어 예측')
    plt.legend()
    plt.title("소비량 예측 결과")
    plt.savefig(PREDICTION_PLOT_PATH)

    return PREDICTION_PLOT_PATH, f"RMSE: {rmse:.2f}"
