# train.py

import os
from data_loader import load_and_process
from material_forecast_model import MaterialForecastModel
from utils import (
    save_scaler,
    plot_learning_curves,
    plot_prediction,
    calculate_metrics,
    needs_retraining
)
from config import (
    MODEL_DIR,
    RESULT_DIR,
    LOG_DIR,
    MAE_THRESHOLD,
    RMSE_THRESHOLD
)

def train_single_fish(fish_name: str, target_column: str, csv_filename: str = None,
                      X=None, y=None, scaler=None):
    print(f"🐟 Start training for: {fish_name} ({target_column})")

    # 1. 데이터 로딩 (직접 입력 X, y, scaler가 없다면 파일에서 로딩)
    if X is None or y is None or scaler is None:
        if csv_filename is None:
            raise ValueError("Either provide (X, y, scaler) or csv_filename.")
        X, y, scaler = load_and_process(csv_filename, target_column)

    input_shape = X.shape[1:] if len(X.shape) == 3 else (X.shape[1],)
    output_days = y.shape[1] if len(y.shape) == 2 else 1

    # 2. 모델 초기화 및 학습
    model = MaterialForecastModel(input_shape=input_shape, output_days=output_days)
    history = model.train(X, y, epochs=30, batch_size=32)

    # 3. 모델 및 스케일러 저장
    model.save(fish_name)
    save_scaler(scaler, fish_name, MODEL_DIR)

    # 4. 예측 수행
    y_pred = model.predict(X)

    # 5. 시각화 디렉토리 준비
    result_dir = os.path.join(RESULT_DIR, fish_name)
    os.makedirs(result_dir, exist_ok=True)

    # 5-1. 학습 곡선 시각화
    plot_learning_curves(history, os.path.join(result_dir, "loss_curve.png"))

    # 5-2. 예측 vs 실제 시각화 (스케일 복원 포함)
    plot_prediction(
        y[:1], y_pred[:1],
        save_path=os.path.join(result_dir, "sample_prediction.png"),
        scaler=scaler,
        target_column=target_column,
        title=f"{fish_name} 예측 결과 (1주일)"
    )

    # 6. 전체 평가
    mae, rmse = calculate_metrics(y, y_pred)
    print(f"[{fish_name}] MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    # 7. 로그 저장
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"{fish_name}.txt")
    with open(log_path, 'w') as f:
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"Needs retraining: {needs_retraining(mae, rmse, MAE_THRESHOLD, RMSE_THRESHOLD)}\n")

    # 8. 재학습 여부 판단
    if needs_retraining(mae, rmse, MAE_THRESHOLD, RMSE_THRESHOLD):
        print(f"🔁 [{fish_name}] 재학습이 필요합니다.")
    else:
        print(f"✅ [{fish_name}] 성능 양호 - 모델 사용 가능.")

    print(f"📦 Training completed for {fish_name}.\n")
