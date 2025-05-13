# main.py
from fastapi import FastAPI, APIRouter, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from train import train_single_fish
from data_loader import preprocess_dataframe
from utils import calculate_metrics, plot_prediction
from material_forecast_model import MaterialForecastModel
import pandas as pd
import base64
import os
import shutil
import uuid
from datetime import datetime
import pytz
from config import RAW_DATA_DIR, RMSE_THRESHOLD, RESULT_DIR, UPLOAD_DIR, INPUT_DAYS, OUTPUT_DAYS, LOG_DIR

app = FastAPI()
router = APIRouter()

# 타임존 설정
timezone = pytz.timezone("Asia/Seoul")

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

# CSV 파일 업로드 및 두 LSTM 모델 결과 처리
import os

@router.post("/upload")
async def post_data_set(file: UploadFile = File(...)):
    try:
        current_time = datetime.now(timezone).strftime("%Y%m%d_%H%M%S")
        new_filename = f"{current_time}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, new_filename)
        raw_copy_path = os.path.join(RAW_DATA_DIR, new_filename)

        # 업로드된 파일 저장
        with open(file_path, "wb") as f:
            f.write(await file.read())
        shutil.copy(file_path, raw_copy_path)

        # CSV 파일 로드
        df = pd.read_csv(file_path)

        # ✅ Date 컬럼 생성 처리 (없을 경우 연도/월/일 기반 생성)
        if 'Date' not in df.columns:
            df = df.rename(columns={'연도': 'year', '월': 'month', '일': 'day'})
            if all(col in df.columns for col in ['year', 'month', 'day']):
                df['Date'] = pd.to_datetime(df[['year', 'month', 'day']])
            else:
                raise HTTPException(status_code=400, detail="Date 컬럼 또는 연도/월/일 컬럼이 존재하지 않습니다.")
        df.set_index('Date', inplace=True)

        # 기본 생선명 및 타겟
        fish_name = "광어"
        target_column = "광어 소비량(g)"

        # 전처리 및 시퀀스 생성
        scaled_df, scaler = preprocess_dataframe(df.copy(), target_column)
        input_data = scaled_df[-INPUT_DAYS:].drop(columns=[target_column]).values.reshape(1, INPUT_DAYS, -1)
        y_true_scaled = scaled_df[-OUTPUT_DAYS:][target_column].values.reshape(1, -1)

        # 예측 (v1)
        model_v1 = MaterialForecastModel.load(fish_name)
        y_pred_scaled_v1 = model_v1.predict(input_data)
        mae_v1, rmse_v1 = calculate_metrics(y_true_scaled, y_pred_scaled_v1)

        # 재학습 조건
        retrained = False
        if rmse_v1 > RMSE_THRESHOLD:
            train_single_fish(fish_name, target_column, new_filename)
            retrained = True

        # 예측 (v2)
        model_v2 = MaterialForecastModel.load(fish_name)
        y_pred_scaled_v2 = model_v2.predict(input_data)
        mae_v2, rmse_v2 = calculate_metrics(y_true_scaled, y_pred_scaled_v2)

        # 로그 저장
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(os.path.join(LOG_DIR, f"{fish_name}.txt"), "w") as log_file:
            log_file.write(f"MAE: {mae_v2:.4f}\n")
            log_file.write(f"RMSE: {rmse_v2:.4f}\n")
            log_file.write(f"Needs retraining: {retrained}\n")

        # 역변환 함수
        def inverse_transform(y_scaled):
            return scaler.inverse_transform(
                np.hstack([
                    np.zeros((OUTPUT_DAYS, scaler.n_features_in_ - 1)),
                    y_scaled.reshape(OUTPUT_DAYS, 1)
                ])
            )[:, -1].reshape(1, -1)

        y_true = inverse_transform(y_true_scaled)
        y_pred_v1 = inverse_transform(y_pred_scaled_v1)
        y_pred_v2 = inverse_transform(y_pred_scaled_v2)

        # 시각화 및 이미지 파일 저장
        result_dir = os.path.join(RESULT_DIR, fish_name)
        os.makedirs(result_dir, exist_ok=True)
        v1_img_path = os.path.join(result_dir, f"v1_{uuid.uuid4().hex[:6]}.png")
        v2_img_path = os.path.join(result_dir, f"v2_{uuid.uuid4().hex[:6]}.png")
        plot_prediction(y_true, y_pred_v1, v1_img_path, title=f"{fish_name} LSTM 예측 (v1)")
        plot_prediction(y_true, y_pred_v2, v2_img_path, title=f"{fish_name} LSTM 예측 (v2)")

        # ✅ 반환 포맷
        return {
            "result_visualizing_LSTM": get_img(v1_img_path),
            "result_evaluating_LSTM": y_pred_v1.tolist(),
            "result_visualizing_LSTM_v2": get_img(v2_img_path),
            "result_evaluating_LSTM_v2": y_pred_v2.tolist(),
            "saved_filename": new_filename
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# CORS 설정
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)