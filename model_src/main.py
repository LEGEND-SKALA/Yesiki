from fastapi import FastAPI, APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
import numpy as np

from config import RAW_DATA_DIR, RESULT_DIR, MODEL_DIR, MAE_THRESHOLD, RMSE_THRESHOLD
from data_loader import load_and_process
from material_forecast_model import MaterialForecastModel
from train import train_single_fish
from utils import (
    load_scaler, save_scaler, calculate_metrics, needs_retraining,
    plot_prediction, get_img
)

app = FastAPI()
router = APIRouter()

targets = {
    "광어": "광어 소비량(g)",
    "연어": "연어 소비량(g)",
    "장어": "장어 소비량(g)"
}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        filename = file.filename
        file_path = os.path.join(RAW_DATA_DIR, filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        results = {}

        for fish_name, target_column in targets.items():
            X, y, _ = load_and_process(filename, target_column)
            retrained = False

            try:
                model = MaterialForecastModel.load(fish_name)
                scaler = load_scaler(fish_name, MODEL_DIR)
            except:
                train_single_fish(fish_name, target_column, filename)
                retrained = True
                model = MaterialForecastModel.load(fish_name)
                scaler = load_scaler(fish_name, MODEL_DIR)

            y_pred = model.predict(X)
            mae, rmse = calculate_metrics(y, y_pred)

            if needs_retraining(mae, rmse, MAE_THRESHOLD, RMSE_THRESHOLD):
                train_single_fish(fish_name, target_column, filename)
                retrained = True
                model = MaterialForecastModel.load(fish_name)
                scaler = load_scaler(fish_name, MODEL_DIR)
                y_pred = model.predict(X)

            # 이미지 저장 및 인코딩
            result_dir = os.path.join(RESULT_DIR, fish_name)
            os.makedirs(result_dir, exist_ok=True)
            img_path = os.path.join(result_dir, "sample_prediction_api.png")
            plot_prediction(y[:1], y_pred[:1], img_path, scaler, target_column)
            encoded_img = get_img(img_path)

            # 예측 총합
            y_pred_rescaled = scaler.inverse_transform(
                np.hstack([np.zeros((y_pred.shape[1], scaler.n_features_in_ - 1)), y_pred[0].reshape(-1, 1)])
            )[:, -1]
            total_prediction = int(np.sum(y_pred_rescaled))

            # 응답 결과 키로 삽입
            results[f"{fish_name}_결과_이미지"] = encoded_img
            results[f"{fish_name}_1주_발주량 예측"] = total_prediction
            results[f"{fish_name}_재학습_여부"] = int(retrained)

        return JSONResponse(content=results)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))  # 404 Not Found 반환

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))  # 500 Internal Server Error 반환

