"""
Optuna 하이퍼파라미터 튜닝
- 최적 LightGBM 파라미터 찾기
- 50 trials
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
import joblib
from tqdm import tqdm
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')

# 수정된 feature extractor 사용
from feature_extractor_v3 import FeatureExtractorV3


def load_train_csv():
    """train.csv 로드"""
    train_path = "D:/workspce/kleague_project/data/train.csv"
    print("Loading train.csv...")
    df_train = pd.read_csv(train_path)
    print(f"✅ Loaded {len(df_train):,} rows")
    return df_train


def prepare_training_data_v2(df_train):
    """학습 데이터 준비"""
    print("\nPreparing training data...")
    
    extractor = FeatureExtractorV3()
    features_list = []
    labels_list = []
    failed_count = 0
    
    grouped = df_train.groupby('game_episode')
    
    for game_episode, df_episode in tqdm(grouped, desc="Processing episodes"):
        try:
            df_episode = df_episode.sort_values('action_id').reset_index(drop=True)
            
            # 타겟 찾기
            pass_actions = df_episode[df_episode['type_name'] == 'Pass']
            
            if len(pass_actions) == 0:
                failed_count += 1
                continue
            
            last_pass = pass_actions.iloc[-1]
            target_x = last_pass['end_x']
            target_y = last_pass['end_y']
            
            if pd.isna(target_x) or pd.isna(target_y):
                valid_actions = df_episode.dropna(subset=['end_x', 'end_y'])
                if len(valid_actions) > 0:
                    last_valid = valid_actions.iloc[-1]
                    target_x = last_valid['end_x']
                    target_y = last_valid['end_y']
                    last_pass = last_valid
                else:
                    failed_count += 1
                    continue
            
            # Feature 추출
            last_pass_idx = last_pass.name
            df_for_features = df_episode[df_episode.index < last_pass_idx].copy()
            
            last_pass_row = last_pass.copy()
            last_pass_row['end_x'] = np.nan
            last_pass_row['end_y'] = np.nan
            
            df_with_last = pd.concat([df_for_features, last_pass_row.to_frame().T], ignore_index=True)
            
            features = extractor.extract_features(df_with_last)
            features['game_episode'] = game_episode
            features['game_id'] = last_pass['game_id']
            features_list.append(features)
            
            labels_list.append({
                'game_episode': game_episode,
                'target_x': target_x,
                'target_y': target_y
            })
            
        except Exception as e:
            failed_count += 1
            continue
    
    df_features = pd.DataFrame(features_list)
    df_labels = pd.DataFrame(labels_list)
    df_final = df_features.merge(df_labels, on='game_episode')
    
    print(f"✅ Processed: {len(df_final):,} episodes")
    return df_final


def objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective 함수"""
    
    # 하이퍼파라미터 탐색 범위
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 32, 128),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'lambda_l1': trial.suggest_float('lambda_l1', 0, 10),
        'lambda_l2': trial.suggest_float('lambda_l2', 0, 10),
        'verbose': -1,
        'random_state': 42
    }
    
    # X 좌표 학습
    train_data = lgb.Dataset(X_train, label=y_train['target_x'])
    val_data = lgb.Dataset(X_val, label=y_val['target_x'], reference=train_data)
    
    model_x = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=30)]
    )
    
    pred_x = model_x.predict(X_val)
    
    # Y 좌표 학습
    train_data = lgb.Dataset(X_train, label=y_train['target_y'])
    val_data = lgb.Dataset(X_val, label=y_val['target_y'], reference=train_data)
    
    model_y = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=30)]
    )
    
    pred_y = model_y.predict(X_val)
    
    # 유클리드 거리 계산
    pred_x = np.clip(pred_x, 0, 105)
    pred_y = np.clip(pred_y, 0, 68)
    
    distances = np.sqrt((pred_x - y_val['target_x'])**2 + (pred_y - y_val['target_y'])**2)
    
    return distances.mean()


def tune_hyperparameters(df_train, n_trials=50):
    """하이퍼파라미터 튜닝"""
    print("\n" + "="*70)
    print("🔍 Optuna Hyperparameter Tuning")
    print("="*70)
    
    # Feature 컬럼
    feature_cols = [col for col in df_train.columns 
                   if col not in ['game_episode', 'target_x', 'target_y', 'game_id']]
    
    # Train/Val split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(df_train, groups=df_train['game_id']))
    
    df_train_split = df_train.iloc[train_idx]
    df_val_split = df_train.iloc[val_idx]
    
    X_train = df_train_split[feature_cols]
    y_train = df_train_split[['target_x', 'target_y']]
    X_val = df_val_split[feature_cols]
    y_val = df_val_split[['target_x', 'target_y']]
    
    print(f"\nTrain: {len(X_train):,} episodes")
    print(f"Val: {len(X_val):,} episodes")
    print(f"\nRunning {n_trials} trials...")
    
    # Optuna study
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42)
    )
    
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    print("\n" + "="*70)
    print("✅ Tuning Completed!")
    print("="*70)
    print(f"\n🎯 Best Score: {study.best_value:.4f}")
    print(f"\n📊 Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    return study.best_params


def train_with_best_params(df_train, best_params):
    """최적 파라미터로 학습"""
    print("\n" + "="*70)
    print("🚀 Training with Best Parameters")
    print("="*70)
    
    # Feature 컬럼
    feature_cols = [col for col in df_train.columns 
                   if col not in ['game_episode', 'target_x', 'target_y', 'game_id']]
    
    # Train/Val split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(df_train, groups=df_train['game_id']))
    
    df_train_split = df_train.iloc[train_idx]
    df_val_split = df_train.iloc[val_idx]
    
    X_train = df_train_split[feature_cols]
    y_train_x = df_train_split['target_x']
    y_train_y = df_train_split['target_y']
    X_val = df_val_split[feature_cols]
    y_val_x = df_val_split['target_x']
    y_val_y = df_val_split['target_y']
    
    # 최적 파라미터 적용
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'random_state': 42,
        **best_params
    }
    
    print("\n🔵 Training X coordinate model...")
    train_data_x = lgb.Dataset(X_train, label=y_train_x)
    val_data_x = lgb.Dataset(X_val, label=y_val_x, reference=train_data_x)
    
    model_x = lgb.train(
        params,
        train_data_x,
        num_boost_round=1000,
        valid_sets=[train_data_x, val_data_x],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )
    
    print("\n🔴 Training Y coordinate model...")
    train_data_y = lgb.Dataset(X_train, label=y_train_y)
    val_data_y = lgb.Dataset(X_val, label=y_val_y, reference=train_data_y)
    
    model_y = lgb.train(
        params,
        train_data_y,
        num_boost_round=1000,
        valid_sets=[train_data_y, val_data_y],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )
    
    # 검증
    pred_x = model_x.predict(X_val)
    pred_y = model_y.predict(X_val)
    pred_x = np.clip(pred_x, 0, 105)
    pred_y = np.clip(pred_y, 0, 68)
    
    distances = np.sqrt((pred_x - y_val_x)**2 + (pred_y - y_val_y)**2)
    
    print("\n" + "="*70)
    print("📊 Validation Results")
    print("="*70)
    print(f"Average Euclidean Distance: {distances.mean():.4f}")
    print("="*70)
    
    return model_x, model_y, feature_cols, distances.mean()


def save_models(model_x, model_y, feature_cols, val_score, best_params):
    """모델 저장"""
    output_dir = Path("D:/workspce/kleague_project/models")
    output_dir.mkdir(exist_ok=True)
    
    model_x.save_model(str(output_dir / 'lgbm_tuned_x.txt'))
    model_y.save_model(str(output_dir / 'lgbm_tuned_y.txt'))
    joblib.dump(feature_cols, output_dir / 'feature_cols_tuned.pkl')
    joblib.dump(best_params, output_dir / 'best_params.pkl')
    
    meta = {
        'val_score': val_score,
        'best_params': best_params,
        'num_features': len(feature_cols)
    }
    joblib.dump(meta, output_dir / 'model_meta_tuned.pkl')
    
    print(f"\n💾 Models saved to: {output_dir}/")


if __name__ == "__main__":
    print("="*70)
    print("🎯 Optuna Hyperparameter Tuning + Training")
    print("="*70)
    
    # 1. 데이터 로드
    df_train_raw = load_train_csv()
    
    # 2. 데이터 준비
    df_train = prepare_training_data_v2(df_train_raw)
    
    # 3. 하이퍼파라미터 튜닝
    best_params = tune_hyperparameters(df_train, n_trials=50)
    
    # 4. 최적 파라미터로 학습
    model_x, model_y, feature_cols, val_score = train_with_best_params(df_train, best_params)
    
    # 5. 저장
    save_models(model_x, model_y, feature_cols, val_score, best_params)
    
    print("\n✅ Completed!")
    print(f"🎯 Validation Score: {val_score:.4f}")
    print("\n📝 Next: Run inference with tuned models")