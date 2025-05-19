# material_forecast_model.py

import os
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from config import MODEL_DIR

class MaterialForecastModel:
    def __init__(self, input_shape, output_days):
        """모델 구조 정의"""
        self.model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(units=output_days)  # 7일 예측
        ])
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    def train(self, X_train, y_train, epochs=30, batch_size=32, validation_split=0.1, patience=5):
        """모델 학습"""
        early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=1
        )
        return history

    def predict(self, X):
        """예측"""
        return self.model.predict(X)

    def save(self, fish_name):
        """모델 저장"""
        model_path = os.path.join(MODEL_DIR, fish_name, "model.h5")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)

    @staticmethod
    def load(fish_name):
        """모델 로딩"""
        model_path = os.path.join(MODEL_DIR, fish_name, "model.h5")
        model = load_model(model_path,compile=False)
        return model
