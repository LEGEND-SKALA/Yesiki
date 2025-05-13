# data_loader.py
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, INPUT_DAYS, OUTPUT_DAYS

def load_raw_data(filename: str) -> pd.DataFrame:
    """원본 CSV 파일 로딩 후 날짜 컬럼 생성"""
    filepath = os.path.join(RAW_DATA_DIR, filename)
    df = pd.read_csv(filepath)

    # 날짜 컬럼 생성
    df['Date'] = pd.to_datetime(df[['연도', '월', '일']])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)

    return df


def preprocess_dataframe(df: pd.DataFrame, target_column: str) -> tuple:
    """
    타겟 소비량 열을 포함한 전체 피처 스케일링
    타겟 열을 맨 뒤로 이동
    """
    features = [
        '기온 (°C)', '강수량 (mm)', '점심 피크타임 손님 수', '저녁 피크타임 손님 수',
        '대인 손님 수', '소인 손님 수', '테이블 회전률',
        target_column
    ]

    df = df[features].copy()

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    return pd.DataFrame(scaled_data, index=df.index, columns=features), scaler


def make_sequences(df: pd.DataFrame, input_days: int, output_days: int, target_column: str):
    """
    시계열 학습을 위한 (X, y) 시퀀스 생성
    X: (samples, input_days, features)
    y: (samples, output_days)
    """
    data = df.values
    target_idx = df.columns.get_loc(target_column)

    X, y = [], []
    for i in range(len(data) - input_days - output_days + 1):
        X.append(data[i:i + input_days])
        y.append(data[i + input_days:i + input_days + output_days, target_idx])

    X = np.array(X)
    y = np.array(y)

    return X, y


def load_and_process(filename: str, target_column: str):
    """전체 파이프라인 실행: 데이터 로딩 → 정규화 → 시퀀스 생성"""
    raw_df = load_raw_data(filename)
    scaled_df, scaler = preprocess_dataframe(raw_df, target_column)
    X, y = make_sequences(scaled_df, INPUT_DAYS, OUTPUT_DAYS, target_column)
    return X, y, scaler
