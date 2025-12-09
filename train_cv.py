"""
Cross Validation + Optuna 하이퍼파라미터 튜닝
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from tqdm import tqdm
import joblib
import optuna
from optuna.samplers import TPESampler

from src.data_loader import DataLoader
from src.features import FeatureExtractor
from src.config import OUTPUT_DIR, MODEL_DIR
from src.utils import setup_logger, set_seed

logger = setup_logger(__name__)

# Optuna 로그 레벨 조정
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_train_data(data_dir: Path):
    """Train 데이터를 로드합니다."""
    logger.info("Train 데이터 로딩 중...")
    
    train_file = data_dir / 'train.csv'
    
    if not train_file.exists():
        logger.error(f"Train 파일을 찾을 수 없습니다: {train_file}")
        raise FileNotFoundError(f"Train 파일이 없습니다: {train_file}")
    
    logger.info(f"✓ Train 파일 발견: {train_file}")
    train_df = pd.read_csv(train_file)
    
    logger.info(f"✓ Train 데이터 로드 완료: {len(train_df):,} rows")
    logger.info(f"  고유 에피소드 수: {train_df['game_episode'].nunique():,}")
    
    return train_df


def extract_train_labels(train_df: pd.DataFrame):
    """Train 데이터에서 정답 레이블을 추출합니다."""
    logger.info("정답 레이블 추출 중...")
    
    labels = []
    
    for game_episode in tqdm(train_df['game_episode'].unique(), desc="레이블 추출"):
        episode_data = train_df[train_df['game_episode'] == game_episode]
        passes = episode_data[episode_data['type_name'] == 'Pass']
        
        if len(passes) > 0:
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
    """Feature를 추출하고 학습 데이터를 준비합니다."""
    logger.info("Feature 추출 중...")
    
    extractor = FeatureExtractor()
    features_list = []
    
    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Feature 추출"):
        game_episode = row['game_episode']
        episode_data = train_df[train_df['game_episode'] == game_episode]
        
        # 마지막 패스 제외
        passes = episode_data[episode_data['type_name'] == 'Pass']
        if len(passes) > 1:
            last_pass_idx = passes.index[-1]
            episode_data_without_last = episode_data[episode_data.index < last_pass_idx]
        else:
            episode_data_without_last = episode_data
        
        features = extractor.extract_features(episode_data_without_last)
        features['game_episode'] = game_episode
        features_list.append(features)
    
    features_df = pd.DataFrame(features_list)
    data = features_df.merge(labels_df, on='game_episode', how='inner')
    
    feature_cols = extractor.get_feature_names()
    X = data[feature_cols].values
    y_x = data['end_x'].values
    y_y = data['end_y'].values
    
    logger.info(f"✓ Feature 추출 완료")
    logger.info(f"  Shape: X={X.shape}, y_x={y_x.shape}, y_y={y_y.shape}")
    
    return X, y_x, y_y, feature_cols


def objective_x(trial, X_train, y_train, X_val, y_val, feature_cols):
    """X 좌표 모델의 Optuna objective 함수"""
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
    }
    
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
    )
    
    pred = model.predict(X_val)
    rmse = np.sqrt(np.mean((pred - y_val) ** 2))
    
    return rmse


def tune_hyperparameters(X, y_x, y_y, feature_cols, n_trials=50):
    """Optuna로 하이퍼파라미터를 튜닝합니다."""
    logger.info(f"Optuna 하이퍼파라미터 튜닝 시작 ({n_trials} trials)...")
    
    # Train/Val split (튜닝용)
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    split = int(0.8 * n_samples)
    
    train_idx = indices[:split]
    val_idx = indices[split:]
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_x_train, y_x_val = y_x[train_idx], y_x[val_idx]
    y_y_train, y_y_val = y_y[train_idx], y_y[val_idx]
    
    # X 좌표 튜닝
    logger.info("X 좌표 모델 하이퍼파라미터 튜닝 중...")
    study_x = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
    study_x.optimize(
        lambda trial: objective_x(trial, X_train, y_x_train, X_val, y_x_val, feature_cols),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    best_params_x = study_x.best_params
    logger.info(f"✓ X 최적 파라미터: RMSE={study_x.best_value:.4f}")
    
    # Y 좌표는 X와 동일한 파라미터 사용 (시간 절약)
    logger.info("Y 좌표 모델은 X와 동일한 파라미터 사용")
    best_params_y = best_params_x.copy()
    
    # 기본 파라미터 추가
    base_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
    }
    
    best_params_x.update(base_params)
    best_params_y.update(base_params)
    
    return best_params_x, best_params_y


def train_with_cv(X, y_x, y_y, feature_cols, params_x, params_y, n_folds=5):
    """5-Fold Cross Validation으로 모델을 학습합니다."""
    logger.info(f"{n_folds}-Fold Cross Validation 학습 시작...")
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    models_x = []
    models_y = []
    
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        logger.info(f"\n=== Fold {fold}/{n_folds} ===")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_x_train, y_x_val = y_x[train_idx], y_x[val_idx]
        y_y_train, y_y_val = y_y[train_idx], y_y[val_idx]
        
        # X 모델 학습
        train_data_x = lgb.Dataset(X_train, label=y_x_train, feature_name=feature_cols)
        val_data_x = lgb.Dataset(X_val, label=y_x_val, feature_name=feature_cols, reference=train_data_x)
        
        model_x = lgb.train(
            params_x,
            train_data_x,
            num_boost_round=1000,
            valid_sets=[val_data_x],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
        )
        
        models_x.append(model_x)
        
        # Y 모델 학습
        train_data_y = lgb.Dataset(X_train, label=y_y_train, feature_name=feature_cols)
        val_data_y = lgb.Dataset(X_val, label=y_y_val, feature_name=feature_cols, reference=train_data_y)
        
        model_y = lgb.train(
            params_y,
            train_data_y,
            num_boost_round=1000,
            valid_sets=[val_data_y],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
        )
        
        models_y.append(model_y)
        
        # Fold 성능 평가
        pred_x = model_x.predict(X_val)
        pred_y = model_y.predict(X_val)
        
        distances = np.sqrt((pred_x - y_x_val)**2 + (pred_y - y_y_val)**2)
        fold_score = distances.mean()
        
        cv_scores.append(fold_score)
        logger.info(f"Fold {fold} Score: {fold_score:.4f}")
    
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)
    
    logger.info(f"\n✓ Cross Validation 완료!")
    logger.info(f"  평균 Score: {mean_score:.4f} (±{std_score:.4f})")
    logger.info(f"  각 Fold: {[f'{s:.4f}' for s in cv_scores]}")
    
    return models_x, models_y, mean_score


def save_cv_models(models_x, models_y, feature_cols, cv_score):
    """CV 모델들을 저장합니다."""
    logger.info("모델 저장 중...")
    
    # 각 Fold 모델 저장
    for i, (model_x, model_y) in enumerate(zip(models_x, models_y), 1):
        model_x.save_model(str(MODEL_DIR / f'lgbm_cv_model_x_fold{i}.txt'))
        model_y.save_model(str(MODEL_DIR / f'lgbm_cv_model_y_fold{i}.txt'))
    
    # Feature 컬럼 저장
    joblib.dump(feature_cols, MODEL_DIR / 'feature_columns_cv.pkl')
    
    # CV Score 저장
    cv_info = {
        'n_folds': len(models_x),
        'cv_score': cv_score,
    }
    joblib.dump(cv_info, MODEL_DIR / 'cv_info.pkl')
    
    logger.info(f"✓ 모델 저장 완료:")
    logger.info(f"  {len(models_x)} Fold 모델")
    logger.info(f"  CV Score: {cv_score:.4f}")


def main():
    """메인 함수"""
    print("=" * 80)
    print("Cross Validation + Optuna 하이퍼파라미터 튜닝")
    print("=" * 80)
    
    set_seed(42)
    
    data_dir = Path('data')
    
    # 1. 데이터 로드
    train_df = load_train_data(data_dir)
    
    # 2. 레이블 추출
    labels_df = extract_train_labels(train_df)
    
    # 3. Feature 추출
    X, y_x, y_y, feature_cols = prepare_training_data(train_df, labels_df)
    
    # 4. 하이퍼파라미터 튜닝
    best_params_x, best_params_y = tune_hyperparameters(X, y_x, y_y, feature_cols, n_trials=50)
    
    # 5. Cross Validation 학습
    models_x, models_y, cv_score = train_with_cv(X, y_x, y_y, feature_cols, best_params_x, best_params_y, n_folds=5)
    
    # 6. 모델 저장
    save_cv_models(models_x, models_y, feature_cols, cv_score)
    
    print("\n" + "=" * 80)
    print("✓ 학습 완료!")
    print("=" * 80)
    print(f"\n최종 CV Score: {cv_score:.4f}")
    print(f"\n이전 모델: 16.08")
    print(f"예상 개선: {((16.08 - cv_score) / 16.08 * 100):.1f}% ↓")
    print("\n다음 단계:")
    print("  1. inference_cv.py 실행하여 테스트 예측")
    print("  2. 제출 파일 생성 및 Dacon 제출")
    print("=" * 80)


if __name__ == "__main__":
    main()