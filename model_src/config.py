import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))

ARTIFACT_DIR = os.path.join(ROOT_DIR, 'artifacts')
DATA_DIR = os.path.join(ARTIFACT_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
UPLOAD_DIR = os.path.join(DATA_DIR, "upload")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

MODEL_DIR = os.path.join(ARTIFACT_DIR, 'models')
RESULT_DIR = os.path.join(ARTIFACT_DIR, 'results')
LOG_DIR = os.path.join(ARTIFACT_DIR, 'logs')

# 모델 및 데이터 관련 상수
INPUT_DAYS = 30
OUTPUT_DAYS = 7
MAE_THRESHOLD = 300
RMSE_THRESHOLD = 500
