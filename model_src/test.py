import os
import sys
import numpy as np
import traceback

from config import RAW_DATA_DIR, UPLOAD_DATA_DIR, RESULT_DIR, MODEL_DIR, MAE_THRESHOLD, RMSE_THRESHOLD
from data_loader import load_and_process_new_data, load_combined_data, get_latest_input_from_raw
from material_forecast_model import MaterialForecastModel
from train import train_single_fish
from utils import (
    load_scaler, save_scaler, calculate_metrics, needs_retraining,
    plot_prediction, get_img, inverse_transform_target_only
)
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

# Define targets
targets = {
    "광어": "광어 소비량(g)",
    "연어": "연어 소비량(g)",
    "장어": "장어 소비량(g)"
}

def main(upload_filename):
    try:
        results = {}
        upload_path = os.path.join(UPLOAD_DATA_DIR, upload_filename)

        for fish_name, target_column in targets.items():
            print(f"\n🔍 Processing {fish_name}...")

            # 1. Load 7-day uploaded data (no sequence generation)
            try:
                X_true_data, y_true_data, _ = load_and_process_new_data(upload_path, target_column)
            except Exception as e:
                results[f"{fish_name}_오류"] = f"데이터 로딩 실패: {str(e)}"
                continue

            # 2. Load model & recent data & scaler
            raw_file_path = os.path.join(RAW_DATA_DIR, "qooqoo_dummy_v0.12.csv")

            try:
                model = MaterialForecastModel.load(fish_name)
                X_input, scaler = get_latest_input_from_raw(raw_file_path, target_column)
            except Exception as e:
                print(f"⚠️ Model not found for {fish_name}. Skipping.")
                results[f"{fish_name}_오류"] = str(e)
                continue

            # 3. Predict
            y_pred = model.predict(X_input)[0]
            mae, rmse = calculate_metrics(y_true_data, y_pred)
            print(f"📊 MAE: {mae:.2f}, RMSE: {rmse:.2f}")

            retrained = False

            # 4. Check if retraining is needed
            if needs_retraining(mae, rmse, MAE_THRESHOLD, RMSE_THRESHOLD):
                print(f"🔁 Retraining {fish_name} using combined 90-day dataset...")
                retrained = True

                X_finetune, y_finetune, scaler = load_combined_data(
                    raw_file_path, upload_path, target_column
                )

                # 🔁 Train again (fine-tune style)
                train_single_fish(fish_name, target_column, raw_file_path, X=X_finetune, y=y_finetune, scaler=scaler)

                # Reload model and predict again
                model = MaterialForecastModel.load(fish_name)
                X_input, scaler = get_latest_input_from_raw(raw_file_path, target_column)
                y_pred = model.predict(X_input)
                retrained = True

            # 5. Plot prediction image
            result_dir = os.path.join(RESULT_DIR, fish_name)
            os.makedirs(result_dir, exist_ok=True)
            img_path = os.path.join(result_dir, "sample_prediction_debug.png")
            plot_prediction(y_true_data[:1], y_pred[:1], img_path, scaler, target_column)
            encoded_img = get_img(img_path)

            # 6. Inverse transform for total prediction
            y_pred_rescaled = inverse_transform_target_only(y_pred, scaler)
            total_prediction = int(np.sum(y_pred_rescaled))


            # Store results
            results[f"{fish_name}_결과_이미지"] = encoded_img
            results[f"{fish_name}_1주_발주량 예측"] = total_prediction
            results[f"{fish_name}_재학습_여부"] = int(retrained)

        # 7. Generate LLM report
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

        line 기준으로 10줄 이내로 작성해줘
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", "{input}")
        ])
        llm = ChatOpenAI(temperature=0)
        chain = prompt | llm
        response = chain.invoke({"input": user_input})

        results["보고서"] = response.content

        print("\n📝 Generated Report:\n")
        print(response.content)

    except Exception as e:
        print("❌ Error occurred:")
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python debug_predict.py <csv_file>")
        sys.exit(1)

    input_csv = sys.argv[1]
    main(input_csv)
