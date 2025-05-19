from fastapi import FastAPI, APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
import numpy as np
import traceback

from config import RAW_DATA_DIR, RESULT_DIR, MODEL_DIR, MAE_THRESHOLD, RMSE_THRESHOLD
from data_loader import load_and_process
from material_forecast_model import MaterialForecastModel
from train import train_single_fish
from utils import (
    load_scaler, save_scaler, calculate_metrics, needs_retraining,
    plot_prediction, get_img
)
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


load_dotenv("D:/workspace/fastapi/AIOps/project/Yesiki/env_sample.env")


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
        # 예시로 단순 보고서 텍스트 생성 (원하는 내용으로 대체 가능)
        system_message = (
            "너는 데이터모델을 사용해서 이후 7일 재료 발주량을 예측하고 "
            "그 결과에 대해서 보고서 형태로 출력하는 업무를 수행해. "
            "보고서에는 각 재료별 일자별 발주량 예측치를 표 형식 또는 정리된 텍스트 형식으로 제시하고, "
            "추세나 특이사항이 있다면 간단한 코멘트도 포함해줘."
        )
        user_input = f"""
        다음은 3가지 재료(광어, 연어, 장어)의 1주일(7일) 간 발주량 예측 결과입니다:

        광어_1주_발주량 예측:
        {results["광어_1주_발주량 예측"]}

        연어_1주_발주량 예측:
        {results["연어_1주_발주량 예측"]}

        장어_1주_발주량 예측:
        {results["장어_1주_발주량 예측"]}

        이 데이터를 바탕으로 보고서를 작성해줘.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", "{input}")
        ])

        llm = ChatOpenAI(temperature=0)

        chain = prompt | llm

        response = chain.invoke({
            "input": user_input
        })

        results["보고서"] = response.content
        return JSONResponse(content=results)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))  # 404 Not Found 반환

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))  # 500 Internal Server Error 반환

# CORS 설정
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)