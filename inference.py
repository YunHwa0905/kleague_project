"""
LightGBM 모델을 사용한 추론 스크립트
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from datetime import datetime
from tqdm import tqdm

from src.data_loader import DataLoader
from src.features import FeatureExtractor
from src.config import DATA_DIR, MODEL_DIR, OUTPUT_DIR
from src.utils import setup_logger

logger = setup_logger(__name__)


def load_models():
    """
    학습된 LightGBM 모델을 로드합니다.
    
    Returns:
        model_x: X 좌표 예측 모델
        model_y: Y 좌표 예측 모델
        feature_cols: Feature 컬럼 이름
    """
    logger.info("모델 로딩 중...")
    
    model_x_path = MODEL_DIR / 'lgbm_model_x.txt'
    model_y_path = MODEL_DIR / 'lgbm_model_y.txt'
    feature_cols_path = MODEL_DIR / 'feature_columns.pkl'
    
    # 파일 존재 확인
    if not model_x_path.exists():
        raise FileNotFoundError(f"X 모델을 찾을 수 없습니다: {model_x_path}")
    if not model_y_path.exists():
        raise FileNotFoundError(f"Y 모델을 찾을 수 없습니다: {model_y_path}")
    if not feature_cols_path.exists():
        raise FileNotFoundError(f"Feature 컬럼을 찾을 수 없습니다: {feature_cols_path}")
    
    # 모델 로드
    model_x = lgb.Booster(model_file=str(model_x_path))
    model_y = lgb.Booster(model_file=str(model_y_path))
    feature_cols = joblib.load(feature_cols_path)
    
    logger.info(f"✓ 모델 로드 완료")
    logger.info(f"  X 모델: {model_x.num_trees()} trees")
    logger.info(f"  Y 모델: {model_y.num_trees()} trees")
    logger.info(f"  Features: {len(feature_cols)} 개")
    
    return model_x, model_y, feature_cols


def predict_with_lgbm(test_df, model_x, model_y, feature_cols):
    """
    LightGBM 모델로 테스트 데이터를 예측합니다.
    
    Args:
        test_df: 테스트 데이터
        model_x: X 좌표 예측 모델
        model_y: Y 좌표 예측 모델
        feature_cols: Feature 컬럼 이름
    
    Returns:
        predictions: DataFrame with game_episode, end_x, end_y
    """
    logger.info("예측 시작...")
    
    extractor = FeatureExtractor()
    
    predictions = []
    
    for game_episode, episode_data in tqdm(test_df.groupby('game_episode'), desc="예측 중"):
        # Feature 추출
        features = extractor.extract_features(episode_data)
        
        # Feature를 올바른 순서로 배열
        X = np.array([features[col] for col in feature_cols]).reshape(1, -1)
        
        # 예측
        pred_x = model_x.predict(X)[0]
        pred_y = model_y.predict(X)[0]
        
        # 좌표 범위 제한
        pred_x = np.clip(pred_x, 0, 105)
        pred_y = np.clip(pred_y, 0, 68)
        
        predictions.append({
            'game_episode': game_episode,
            'end_x': pred_x,
            'end_y': pred_y
        })
    
    predictions_df = pd.DataFrame(predictions)
    
    logger.info(f"✓ 예측 완료: {len(predictions_df)} 개 에피소드")
    
    return predictions_df


def create_submission(predictions_df, sample_submission_path, output_path):
    """
    제출 파일을 생성합니다.
    
    Args:
        predictions_df: 예측 결과
        sample_submission_path: 샘플 제출 파일 경로
        output_path: 출력 파일 경로
    """
    logger.info("제출 파일 생성 중...")
    
    # 샘플 제출 파일 로드
    sample_df = pd.read_csv(sample_submission_path)
    
    # 예측 결과와 병합
    submission = sample_df[['game_episode']].merge(
        predictions_df,
        on='game_episode',
        how='left'
    )
    
    # NaN 처리 (혹시 모를 경우를 대비)
    submission['end_x'].fillna(52.5, inplace=True)
    submission['end_y'].fillna(34.0, inplace=True)
    
    # 저장
    submission.to_csv(output_path, index=False, encoding='utf-8')
    
    logger.info(f"✓ 제출 파일 저장: {output_path}")
    
    # 통계 출력
    logger.info(f"\n제출 파일 통계:")
    logger.info(f"  총 행 수: {len(submission)}")
    logger.info(f"  end_x 범위: [{submission['end_x'].min():.2f}, {submission['end_x'].max():.2f}]")
    logger.info(f"  end_y 범위: [{submission['end_y'].min():.2f}, {submission['end_y'].max():.2f}]")
    logger.info(f"  end_x 평균: {submission['end_x'].mean():.2f}")
    logger.info(f"  end_y 평균: {submission['end_y'].mean():.2f}")
    
    return submission


def main():
    """메인 함수"""
    print("=" * 80)
    print("LightGBM 모델 추론")
    print("=" * 80)
    
    # 1. 모델 로드
    model_x, model_y, feature_cols = load_models()
    
    # 2. 테스트 데이터 로드
    logger.info("테스트 데이터 로딩 중...")
    loader = DataLoader(DATA_DIR)
    test_df = loader.load_all_test_data()
    logger.info(f"✓ 테스트 데이터 로드 완료: {len(test_df):,} rows")
    
    # 3. 예측
    predictions_df = predict_with_lgbm(test_df, model_x, model_y, feature_cols)
    
    # 4. 제출 파일 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"submission_lgbm_{timestamp}.csv"
    output_path = OUTPUT_DIR / output_filename
    
    sample_submission_path = DATA_DIR / 'sample_submission.csv'
    
    submission = create_submission(predictions_df, sample_submission_path, output_path)
    
    print("\n" + "=" * 80)
    print("✓ 추론 완료!")
    print("=" * 80)
    print(f"\n제출 파일: {output_path}")
    print(f"\n예상 점수:")
    print(f"  Validation: 16.08 (학습 시)")
    print(f"  Public Test: 15~18 예상")
    print(f"\n이전 베이스라인 v2: 27.63")
    print(f"예상 개선율: 35-42% ↓")
    print("=" * 80)
    print(f"\n다음 단계:")
    print(f"  1. Dacon에 {output_filename} 제출")
    print(f"  2. Public Score 확인")
    print(f"  3. 추가 개선 (Feature Engineering, 앙상블 등)")
    print("=" * 80)


if __name__ == "__main__":
    main()