"""
LightGBM 모델 학습 스크립트
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import joblib

from src.data_loader import DataLoader
from src.features import FeatureExtractor
from src.config import OUTPUT_DIR, MODEL_DIR, LGBM_PARAMS
from src.utils import setup_logger, set_seed

logger = setup_logger(__name__)


def load_train_data(data_dir: Path):
    """
    Train 데이터를 로드합니다.
    
    Returns:
        train_df: 전체 train 데이터프레임
    """
    logger.info("Train 데이터 로딩 중...")
    
    # train.csv 파일 경로
    train_file = data_dir / 'train.csv'
    
    if not train_file.exists():
        logger.error(f"Train 파일을 찾을 수 없습니다: {train_file}")
        raise FileNotFoundError(f"Train 파일이 없습니다: {train_file}")
    
    logger.info(f"✓ Train 파일 발견: {train_file}")
    
    # CSV 로드
    train_df = pd.read_csv(train_file)
    
    logger.info(f"✓ Train 데이터 로드 완료: {len(train_df):,} rows")
    logger.info(f"  고유 에피소드 수: {train_df['game_episode'].nunique():,}")
    
    return train_df


def extract_train_labels(train_df: pd.DataFrame):
    """
    Train 데이터에서 정답 레이블(마지막 패스의 end_x, end_y)을 추출합니다.
    
    Args:
        train_df: 전체 train 데이터
    
    Returns:
        labels_df: game_episode, end_x, end_y
    """
    logger.info("정답 레이블 추출 중...")
    
    labels = []
    
    for game_episode in tqdm(train_df['game_episode'].unique(), desc="레이블 추출"):
        episode_data = train_df[train_df['game_episode'] == game_episode]
        
        # 패스만 필터링
        passes = episode_data[episode_data['type_name'] == 'Pass']
        
        if len(passes) > 0:
            # 마지막 패스의 end 좌표
            last_pass = passes.iloc[-1]
            
            if pd.notna(last_pass['end_x']) and pd.notna(last_pass['end_y']):
                labels.append({
                    'game_episode': game_episode,
                    'end_x': last_pass['end_x'],
                    'end_y': last_pass['end_y']
                })
    
    labels_df = pd.DataFrame(labels)
    
    logger.info(f"✓ {len(labels_df)}개 레이블 추출 완료")
    
    return labels_df


def prepare_training_data(train_df: pd.DataFrame, labels_df: pd.DataFrame):
    """
    Feature를 추출하고 학습 데이터를 준비합니다.
    
    Args:
        train_df: 전체 train 데이터
        labels_df: 정답 레이블
    
    Returns:
        X: Feature matrix
        y_x: X 좌표 정답
        y_y: Y 좌표 정답
    """
    logger.info("Feature 추출 중...")
    
    extractor = FeatureExtractor()
    
    features_list = []
    
    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Feature 추출"):
        game_episode = row['game_episode']
        episode_data = train_df[train_df['game_episode'] == game_episode]
        
        # Feature 추출 (마지막 패스 제외)
        # 마지막 패스를 예측해야 하므로, 마지막 패스 이전까지만 사용
        passes = episode_data[episode_data['type_name'] == 'Pass']
        if len(passes) > 1:
            # 마지막 패스 제외
            last_pass_idx = passes.index[-1]
            episode_data_without_last = episode_data[episode_data.index < last_pass_idx]
        else:
            episode_data_without_last = episode_data
        
        features = extractor.extract_features(episode_data_without_last)
        features['game_episode'] = game_episode
        features_list.append(features)
    
    features_df = pd.DataFrame(features_list)
    
    # 레이블과 병합
    data = features_df.merge(labels_df, on='game_episode', how='inner')
    
    # X, y 분리
    feature_cols = extractor.get_feature_names()
    X = data[feature_cols].values
    y_x = data['end_x'].values
    y_y = data['end_y'].values
    
    logger.info(f"✓ Feature 추출 완료")
    logger.info(f"  Shape: X={X.shape}, y_x={y_x.shape}, y_y={y_y.shape}")
    
    return X, y_x, y_y, feature_cols


def train_models(X, y_x, y_y, feature_cols):
    """
    LightGBM 모델을 학습합니다.
    
    Args:
        X: Feature matrix
        y_x: X 좌표 정답
        y_y: Y 좌표 정답
        feature_cols: Feature 이름 리스트
    
    Returns:
        model_x: X 예측 모델
        model_y: Y 예측 모델
    """
    logger.info("모델 학습 시작...")
    
    # Train/Val split
    X_train, X_val, y_x_train, y_x_val = train_test_split(
        X, y_x, test_size=0.2, random_state=42
    )
    _, _, y_y_train, y_y_val = train_test_split(
        X, y_y, test_size=0.2, random_state=42
    )
    
    # X 좌표 모델
    logger.info("X 좌표 모델 학습 중...")
    train_data_x = lgb.Dataset(X_train, label=y_x_train, feature_name=feature_cols)
    val_data_x = lgb.Dataset(X_val, label=y_x_val, feature_name=feature_cols, reference=train_data_x)
    
    model_x = lgb.train(
        LGBM_PARAMS,
        train_data_x,
        num_boost_round=1000,
        valid_sets=[train_data_x, val_data_x],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )
    
    logger.info(f"✓ X 모델 학습 완료 (Best iteration: {model_x.best_iteration})")
    
    # Y 좌표 모델
    logger.info("Y 좌표 모델 학습 중...")
    train_data_y = lgb.Dataset(X_train, label=y_y_train, feature_name=feature_cols)
    val_data_y = lgb.Dataset(X_val, label=y_y_val, feature_name=feature_cols, reference=train_data_y)
    
    model_y = lgb.train(
        LGBM_PARAMS,
        train_data_y,
        num_boost_round=1000,
        valid_sets=[train_data_y, val_data_y],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )
    
    logger.info(f"✓ Y 모델 학습 완료 (Best iteration: {model_y.best_iteration})")
    
    # Validation 성능
    val_pred_x = model_x.predict(X_val)
    val_pred_y = model_y.predict(X_val)
    
    # Euclidean distance
    distances = np.sqrt((val_pred_x - y_x_val)**2 + (val_pred_y - y_y_val)**2)
    mean_distance = distances.mean()
    
    logger.info(f"✓ Validation 평균 거리: {mean_distance:.4f}")
    
    return model_x, model_y


def save_models(model_x, model_y, feature_cols):
    """
    학습된 모델을 저장합니다.
    
    Args:
        model_x: X 예측 모델
        model_y: Y 예측 모델
        feature_cols: Feature 이름 리스트
    """
    logger.info("모델 저장 중...")
    
    # 모델 저장
    model_x_path = MODEL_DIR / 'lgbm_model_x.txt'
    model_y_path = MODEL_DIR / 'lgbm_model_y.txt'
    
    model_x.save_model(str(model_x_path))
    model_y.save_model(str(model_y_path))
    
    # Feature 이름 저장
    feature_cols_path = MODEL_DIR / 'feature_columns.pkl'
    joblib.dump(feature_cols, feature_cols_path)
    
    logger.info(f"✓ 모델 저장 완료:")
    logger.info(f"  {model_x_path}")
    logger.info(f"  {model_y_path}")
    logger.info(f"  {feature_cols_path}")


def main():
    """메인 함수"""
    print("=" * 80)
    print("LightGBM 모델 학습")
    print("=" * 80)
    
    # 시드 설정
    set_seed(42)
    
    # Data 경로
    data_dir = Path('data')
    
    # 1. Train 데이터 로드
    train_df = load_train_data(data_dir)
    
    # 2. 정답 레이블 추출
    labels_df = extract_train_labels(train_df)
    
    # 3. Feature 추출
    X, y_x, y_y, feature_cols = prepare_training_data(train_df, labels_df)
    
    # 4. 모델 학습
    model_x, model_y = train_models(X, y_x, y_y, feature_cols)
    
    # 5. 모델 저장
    save_models(model_x, model_y, feature_cols)
    
    print("\n" + "=" * 80)
    print("✓ 학습 완료!")
    print("=" * 80)
    print(f"\n다음 단계:")
    print(f"  1. inference.py 실행하여 테스트 예측")
    print(f"  2. 제출 파일 생성 및 Dacon 제출")
    print("=" * 80)


if __name__ == "__main__":
    main()