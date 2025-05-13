# main.py
from fastapi import FastAPI, APIRouter, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from server_model import weight_used_model
from server_model import model
import importlib
import pandas as pd
import base64
import os
from datetime import datetime
import pytz
from server_model.config import UPLOAD_DIR, IMAGE_DIR, MODEL_IMG_DIR

app = FastAPI()
router = APIRouter()

# 디렉토리 설정
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(MODEL_IMG_DIR, exist_ok=True)

# 타임존 설정
timezone = pytz.timezone("Asia/Seoul")

# 이미지를 Base64로 인코딩하여 반환
def get_img(img_name):
    if not os.path.exists(img_name):
        print(f"🚨 이미지 파일이 존재하지 않습니다: {img_name}")
        raise HTTPException(status_code=404, detail="Image not found")
    try:
        with open(img_name, "rb") as f:
            img_byte_arr = f.read()
        encoded = base64.b64encode(img_byte_arr)
        return "data:image/png;base64," + encoded.decode('ascii')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading image: {str(e)}")

# CSV 파일 업로드 및 모델 처리
@router.post("/upload")
async def post_data_set(file: UploadFile = File(...)):
    try:
        current_time = datetime.now(timezone).strftime("%Y%m%d_%H%M%S")
        new_filename = f"{current_time}_{file.filename}"
        file_location = os.path.join(UPLOAD_DIR, new_filename)

        # 업로드된 파일 저장
        with open(file_location, "wb") as f:
            f.write(await file.read())

        # CSV 읽기 및 처리
        dataset = pd.read_csv(file_location)
        dataset['Date'] = pd.to_datetime(dataset[['연도', '월', '일']])
        dataset.set_index('Date', inplace=True)
        dataset.sort_index(inplace=True)

        # 모델 처리 (재학습 포함)
        importlib.reload(model)
        result_visualizing_LSTM_v2, result_evaluating_LSTM_v2 = model.process(dataset)

        if not os.path.exists(result_visualizing_LSTM_v2):
            raise HTTPException(status_code=500, detail=f"File not found: {result_visualizing_LSTM_v2}")

        return {
            "result_visualizing_LSTM_v2": get_img(result_visualizing_LSTM_v2),
            "result_evaluating_LSTM_v2": result_evaluating_LSTM_v2,
            "saved_filename": new_filename
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 이미지 다운로드 엔드포인트
@router.get("/download")
async def download():
    try:
        img_name = os.path.join(IMAGE_DIR, weight_used_model.get_stock_png())
        return FileResponse(path=img_name, media_type='application/octet-stream', filename="stock.png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 모델 구조 이미지 다운로드
@router.get("/download_shapes")
async def download_model_architecture_shapes():
    try:
        img_name = os.path.join(IMAGE_DIR, weight_used_model.get_model_shapes_png())
        return FileResponse(path=img_name, media_type='application/octet-stream', filename="model_shapes.png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# HTML로 예측 이미지 표시
@router.get("/view-download")
async def view_downloaded_image():
    try:
        img_name = os.path.join(IMAGE_DIR, weight_used_model.get_stock_png())
        img_base64 = get_img(img_name)
        return HTMLResponse(content=f"""
        <html>
            <body>
                <h1>Downloaded Stock Prediction Image</h1>
                <img src="{img_base64}" alt="Stock Prediction Image" />
            </body>
        </html>
        """)
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
