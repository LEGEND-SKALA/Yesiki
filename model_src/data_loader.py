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

    # 날짜 컬럼 생성: 컬럼명을 영문으로 바꿔서 datetime 변환
    df = df.rename(columns={'연도': 'year', '월': 'month', '일': 'day'})
    df['Date'] = pd.to_datetime(df[['year', 'month', 'day']])
    
    df.sort_values(by='Date', inplace=True)  # 날짜 정렬만 수행
    return df


def preprocess_dataframe(df: pd.DataFrame, target_column: str) -> tuple:
    """
    전체 피처 정규화 + 타겟 열을 분리
    """
    # 입력 피처: 0~17 컬럼
    input_features = df.columns[:18].tolist()
    input_df = df[input_features + [target_column]].copy()

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(input_df)

    # 컬럼 순서 유지
    scaled_df = pd.DataFrame(scaled_data, index=df.index, columns=input_features + [target_column])
    return scaled_df, scaler


def make_sequences(df: pd.DataFrame, input_days: int, output_days: int, target_column: str):
    """
    시계열 학습을 위한 (X, y) 시퀀스 생성
    X: (samples, input_days, features)
    y: (samples, output_days)
    """
    
    # 🔥 X에 들어갈 feature 데이터만 추출 (target 제외)
    feature_df = df.drop(columns=[target_column])
    feature_data = feature_df.values
    target_data = df[target_column].values

    X, y = [], []
    for i in range(len(df) - input_days - output_days + 1):
        X.append(feature_data[i:i + input_days])
        y.append(target_data[i + input_days:i + input_days + output_days])
    X = np.array(X)
    y = np.array(y)

    return X, y


def load_and_process(filename: str, target_column: str):
    """전체 파이프라인 실행: 데이터 로딩 → 정규화 → 시퀀스 생성"""
    raw_df = load_raw_data(filename)
    scaled_df, scaler = preprocess_dataframe(raw_df, target_column)
    X, y = make_sequences(scaled_df, INPUT_DAYS, OUTPUT_DAYS, target_column)
    return X, y, scaler


def load_combined_data(raw_filename: str, upload_filename: str) -> pd.DataFrame:
    """원본 데이터 + 새 업로드 데이터를 병합한 전체 시계열 반환"""
    raw_df = load_raw_data(raw_filename)
    upload_df = load_raw_data(upload_filename)
    combined_df = pd.concat([raw_df, upload_df]).drop_duplicates().sort_values(by='Date')
    return combined_df
