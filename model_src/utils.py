import base64
from fastapi import HTTPException
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os

def plot_learning_curves(history, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Val Loss')
    plt.plot(history.history['mae'], label='Train MAE')
    if 'val_mae' in history.history:
        plt.plot(history.history['val_mae'], label='Val MAE')
    plt.title('Training Loss and MAE')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_prediction(y_true, y_pred, save_path, scaler=None, target_column=None, title="Prediction vs Actual"):
    """
    예측 결과 vs 실제값 시각화
    - scaler: 정규화에 사용된 MinMaxScaler
    - target_column: 복원할 열 이름 (필수)
    """
    y_true = y_true.reshape(1, -1)
    y_pred = y_pred.reshape(1, -1)

    if scaler is not None and target_column is not None:
        # (1, 7) → (7, 1) → inverse_transform 후 다시 (1, 7)
        y_true = scaler.inverse_transform(
            np.hstack([np.zeros((y_true.shape[1], scaler.n_features_in_ - 1)), y_true.T])
        )[:, -1].reshape(1, -1)

        y_pred = scaler.inverse_transform(
            np.hstack([np.zeros((y_pred.shape[1], scaler.n_features_in_ - 1)), y_pred.T])
        )[:, -1].reshape(1, -1)

    days = [f"{i+1}day" for i in range(y_true.shape[1])]

    plt.figure(figsize=(10, 5))
    plt.plot(days, y_true.flatten(), label='Actual')
    plt.plot(days, y_pred.flatten(), label='Predicted')
    plt.title(title)
    plt.xlabel('Day')
    plt.ylabel('Consumption (g)')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()



def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

def needs_retraining(mae, rmse, threshold_mae, threshold_rmse):
    return mae > threshold_mae or rmse > threshold_rmse


def save_scaler(scaler, fish_name, model_dir):
    """
    MinMaxScaler 객체를 저장합니다 (.pkl)
    - scaler: sklearn MinMaxScaler
    - fish_name: 생선 이름 ("광어", "연어", "장어")
    - model_dir: 모델 저장 경로의 루트 디렉토리
    """
    save_path = os.path.join(model_dir, fish_name, "scaler.pkl")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(scaler, save_path)
    print(f"Scaler saved to: {save_path}")


def load_scaler(fish_name, model_dir):
    """
    저장된 scaler를 로딩합니다.
    """
    load_path = os.path.join(model_dir, fish_name, "scaler.pkl")
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Scaler not found for {fish_name} at {load_path}")
    return joblib.load(load_path)

# 이미지를 Base64로 인코딩하여 반환
def get_img(img_name):
    if not os.path.exists(img_name):
        print(f"🚨 이미지 파일이 존재하지 않습니다: {img_name}")  # 디버깅용 로그 추가
        raise HTTPException(status_code=404, detail="Image not found")
    try:
        with open(img_name, "rb") as f:
            img_byte_arr = f.read()
        encoded = base64.b64encode(img_byte_arr)
        return "data:image/png;base64," + encoded.decode('ascii')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading image: {str(e)}")
    
def inverse_transform_target_only(y_pred, scaler):
    """
    Inverse-transforms only the target column using the scaler that was trained on full feature data.

    Parameters:
    - y_pred: shape (7,), the predicted values for the target column (scaled)
    - scaler: the full-feature MinMaxScaler used during training

    Returns:
    - y_pred_rescaled: shape (7,), the inverse-transformed prediction values
    """
    y_pred = y_pred.reshape(1, -1)  # (1, 7)

    # Create dummy input: shape (7, total_features), fill others with zeros
    dummy = np.zeros((y_pred.shape[1], scaler.n_features_in_))
    dummy[:, -1] = y_pred.flatten()  # assuming target column is last

    # Inverse-transform all features
    inversed = scaler.inverse_transform(dummy)

    # Return only the target column
    return inversed[:, -1]  # shape: (7,)
