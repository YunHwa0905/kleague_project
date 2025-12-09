"""
5-Fold CV 모델 앙상블 추론 스크립트
"""

import sys
from pathlib import Path

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


def load_cv_models():
    """
    5-Fold CV 모델들을 로드합니다.
    
    Returns:
        models_x: X 좌표 예측 모델 리스트
        models_y: Y 좌표 예측 모델 리스트
        feature_cols: Feature 컬럼 이름
        cv_info: CV 정보
    """
    logger.info("CV 모델 로딩 중...")
    
    # CV 정보 로드
    cv_info_path = MODEL_DIR / 'cv_info.pkl'
    if not cv_info_path.exists():
        raise FileNotFoundError(f"CV 정보를 찾을 수 없습니다: {cv_info_path}")
    
    cv_info = joblib.load(cv_info_path)
    n_folds = cv_info['n_folds']
    
    # Feature 컬럼 로드
    feature_cols_path = MODEL_DIR / 'feature_columns_cv.pkl'
    if not feature_cols_path.exists():
        raise FileNotFoundError(f"Feature 컬럼을 찾을 수 없습니다: {feature_cols_path}")
    
    feature_cols = joblib.load(feature_cols_path)
    
    # 각 Fold 모델 로드
    models_x = []
    models_y = []
    
    for fold in range(1, n_folds + 1):
        model_x_path = MODEL_DIR / f'lgbm_cv_model_x_fold{fold}.txt'
        model_y_path = MODEL_DIR / f'lgbm_cv_model_y_fold{fold}.txt'
        
        if not model_x_path.exists():
            raise FileNotFoundError(f"X 모델 Fold {fold}를 찾을 수 없습니다: {model_x_path}")
        if not model_y_path.exists():
            raise FileNotFoundError(f"Y 모델 Fold {fold}를 찾을 수 없습니다: {model_y_path}")
        
        model_x = lgb.Booster(model_file=str(model_x_path))
        model_y = lgb.Booster(model_file=str(model_y_path))
        
        models_x.append(model_x)
        models_y.append(model_y)
    
    logger.info(f"✓ {n_folds}-Fold CV 모델 로드 완료")
    logger.info(f"  CV Score: {cv_info['cv_score']:.4f}")
    logger.info(f"  Features: {len(feature_cols)} 개")
    
    return models_x, models_y, feature_cols, cv_info


def predict_with_cv_ensemble(test_df, models_x, models_y, feature_cols):
    """
    5-Fold CV 모델의 앙상블 예측을 수행합니다.
    
    Args:
        test_df: 테스트 데이터
        models_x: X 좌표 예측 모델 리스트
        models_y: Y 좌표 예측 모델 리스트
        feature_cols: Feature 컬럼 이름
    
    Returns:
        predictions: DataFrame with game_episode, end_x, end_y
    """
    logger.info(f"예측 시작 ({len(models_x)}-Fold 앙상블)...")
    
    extractor = FeatureExtractor()
    
    predictions = []
    
    for game_episode, episode_data in tqdm(test_df.groupby('game_episode'), desc="예측 중"):
        # Feature 추출
        features = extractor.extract_features(episode_data)
        
        # 누락된 Feature에 기본값 추가
        for col in feature_cols:
            if col not in features:
                features[col] = 0.0
        
        # Feature를 올바른 순서로 배열
        X = np.array([features[col] for col in feature_cols]).reshape(1, -1)
        
        # 각 Fold 모델로 예측
        pred_x_list = []
        pred_y_list = []
        
        for model_x, model_y in zip(models_x, models_y):
            pred_x = model_x.predict(X)[0]
            pred_y = model_y.predict(X)[0]
            
            pred_x_list.append(pred_x)
            pred_y_list.append(pred_y)
        
        # 앙상블: 평균
        pred_x = np.mean(pred_x_list)
        pred_y = np.mean(pred_y_list)
        
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
    """제출 파일을 생성합니다."""
    logger.info("제출 파일 생성 중...")
    
    sample_df = pd.read_csv(sample_submission_path)
    
    submission = sample_df[['game_episode']].merge(
        predictions_df,
        on='game_episode',
        how='left'
    )
    
    submission['end_x'].fillna(52.5, inplace=True)
    submission['end_y'].fillna(34.0, inplace=True)
    
    submission.to_csv(output_path, index=False, encoding='utf-8')
    
    logger.info(f"✓ 제출 파일 저장: {output_path}")
    
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
    print("5-Fold CV 모델 앙상블 추론")
    print("=" * 80)
    
    # 1. CV 모델 로드
    models_x, models_y, feature_cols, cv_info = load_cv_models()
    
    # 2. 테스트 데이터 로드
    logger.info("테스트 데이터 로딩 중...")
    loader = DataLoader(DATA_DIR)
    test_df = loader.load_all_test_data()
    logger.info(f"✓ 테스트 데이터 로드 완료: {len(test_df):,} rows")
    
    # 3. 앙상블 예측
    predictions_df = predict_with_cv_ensemble(test_df, models_x, models_y, feature_cols)
    
    # 4. 제출 파일 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"submission_cv_{timestamp}.csv"
    output_path = OUTPUT_DIR / output_filename
    
    sample_submission_path = DATA_DIR / 'sample_submission.csv'
    
    submission = create_submission(predictions_df, sample_submission_path, output_path)
    
    print("\n" + "=" * 80)
    print("✓ 추론 완료!")
    print("=" * 80)
    print(f"\n제출 파일: {output_path}")
    print(f"\n성능 비교:")
    print(f"  이전 모델 (Single): 16.08")
    print(f"  현재 모델 (5-Fold CV): {cv_info['cv_score']:.4f}")
    print(f"  개선율: {((16.08 - cv_info['cv_score']) / 16.08 * 100):.1f}% ↓")
    print(f"\n예상 Public Score: 13~15")
    print("=" * 80)
    print(f"\n다음 단계:")
    print(f"  1. Dacon에 {output_filename} 제출")
    print(f"  2. Public Score 확인")
    print("=" * 80)


if __name__ == "__main__":
    main()