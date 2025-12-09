"""
5-Fold CV 앙상블 학습
- 5개 모델 학습
- 각각 다른 validation set
- 앙상블로 안정적인 성능
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.model_selection import KFold
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from feature_extractor_v4 import FeatureExtractorV4


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
    
    extractor = FeatureExtractorV4()
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


def train_5fold_cv(df_train, n_splits=5):
    """5-Fold CV로 학습"""
    print("\n" + "="*70)
    print("🚀 5-Fold Cross Validation Training")
    print("="*70)
    
    # Feature 컬럼
    feature_cols = [col for col in df_train.columns 
                   if col not in ['game_episode', 'target_x', 'target_y', 'game_id']]
    
    print(f"\n📊 Using {len(feature_cols)} features")
    print(f"📊 Total episodes: {len(df_train):,}")
    print(f"📊 Number of folds: {n_splits}")
    
    # game_id로 그룹화하여 Fold 생성
    unique_games = df_train['game_id'].unique()
    print(f"📊 Total games: {len(unique_games)}")
    
    # KFold 설정
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # 최적 파라미터 (Optuna 결과 사용)
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 128,
        'learning_rate': 0.023,
        'feature_fraction': 0.90,
        'bagging_fraction': 0.81,
        'bagging_freq': 4,
        'min_child_samples': 32,
        'lambda_l1': 1.0,
        'lambda_l2': 7.3,
        'verbose': -1,
        'random_state': 42
    }
    
    models_x = []
    models_y = []
    cv_scores = []
    
    # 각 Fold별 학습
    for fold, (train_game_idx, val_game_idx) in enumerate(kf.split(unique_games), 1):
        print("\n" + "="*70)
        print(f"📁 Fold {fold}/{n_splits}")
        print("="*70)
        
        # 게임 ID로 train/val split
        train_games = unique_games[train_game_idx]
        val_games = unique_games[val_game_idx]
        
        df_train_fold = df_train[df_train['game_id'].isin(train_games)]
        df_val_fold = df_train[df_train['game_id'].isin(val_games)]
        
        print(f"Train: {len(df_train_fold):,} episodes from {len(train_games)} games")
        print(f"Val:   {len(df_val_fold):,} episodes from {len(val_games)} games")
        
        # 데이터 준비
        X_train = df_train_fold[feature_cols]
        y_train_x = df_train_fold['target_x']
        y_train_y = df_train_fold['target_y']
        
        X_val = df_val_fold[feature_cols]
        y_val_x = df_val_fold['target_x']
        y_val_y = df_val_fold['target_y']
        
        # X 좌표 모델 학습
        print(f"\n🔵 Training X model (Fold {fold})...")
        train_data_x = lgb.Dataset(X_train, label=y_train_x)
        val_data_x = lgb.Dataset(X_val, label=y_val_x, reference=train_data_x)
        
        model_x = lgb.train(
            params,
            train_data_x,
            num_boost_round=1000,
            valid_sets=[val_data_x],
            valid_names=['val'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=0)  # 출력 최소화
            ]
        )
        
        # Y 좌표 모델 학습
        print(f"🔴 Training Y model (Fold {fold})...")
        train_data_y = lgb.Dataset(X_train, label=y_train_y)
        val_data_y = lgb.Dataset(X_val, label=y_val_y, reference=train_data_y)
        
        model_y = lgb.train(
            params,
            train_data_y,
            num_boost_round=1000,
            valid_sets=[val_data_y],
            valid_names=['val'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=0)  # 출력 최소화
            ]
        )
        
        # 검증
        pred_x = model_x.predict(X_val)
        pred_y = model_y.predict(X_val)
        
        pred_x = np.clip(pred_x, 0, 105)
        pred_y = np.clip(pred_y, 0, 68)
        
        distances = np.sqrt((pred_x - y_val_x)**2 + (pred_y - y_val_y)**2)
        fold_score = distances.mean()
        
        print(f"\n✅ Fold {fold} Score: {fold_score:.4f}")
        
        # 저장
        models_x.append(model_x)
        models_y.append(model_y)
        cv_scores.append(fold_score)
    
    # 전체 CV 결과
    print("\n" + "="*70)
    print("📊 Cross Validation Results")
    print("="*70)
    
    for i, score in enumerate(cv_scores, 1):
        print(f"Fold {i}: {score:.4f}")
    
    print(f"\n🎯 Average CV Score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    print("="*70)
    
    return models_x, models_y, feature_cols, cv_scores


def save_cv_models(models_x, models_y, feature_cols, cv_scores):
    """5-Fold 모델 저장"""
    output_dir = Path("D:/workspce/kleague_project/models")
    output_dir.mkdir(exist_ok=True)
    
    print("\n💾 Saving models...")
    
    # 각 Fold 모델 저장
    for i, (model_x, model_y) in enumerate(zip(models_x, models_y), 1):
        model_x.save_model(str(output_dir / f'lgbm_cv_x_fold{i}.txt'))
        model_y.save_model(str(output_dir / f'lgbm_cv_y_fold{i}.txt'))
    
    # Feature 컬럼과 메타 정보
    joblib.dump(feature_cols, output_dir / 'feature_cols_cv.pkl')
    
    meta = {
        'cv_scores': cv_scores,
        'avg_score': np.mean(cv_scores),
        'std_score': np.std(cv_scores),
        'n_folds': len(cv_scores),
        'num_features': len(feature_cols)
    }
    joblib.dump(meta, output_dir / 'cv_meta.pkl')
    
    print(f"\n✅ Saved {len(models_x)} fold models to: {output_dir}/")
    print(f"  - lgbm_cv_x_fold1~{len(models_x)}.txt")
    print(f"  - lgbm_cv_y_fold1~{len(models_y)}.txt")
    print(f"  - feature_cols_cv.pkl")
    print(f"  - cv_meta.pkl")


if __name__ == "__main__":
    print("="*70)
    print("🎯 5-Fold Cross Validation Ensemble Training")
    print("="*70)
    
    # 1. 데이터 로드
    df_train_raw = load_train_csv()
    
    # 2. 데이터 준비
    df_train = prepare_training_data_v2(df_train_raw)
    
    # 3. 5-Fold CV 학습
    models_x, models_y, feature_cols, cv_scores = train_5fold_cv(df_train, n_splits=5)
    
    # 4. 저장
    save_cv_models(models_x, models_y, feature_cols, cv_scores)
    
    print("\n" + "="*70)
    print("✅ Training Completed Successfully!")
    print("="*70)
    print(f"\n🎯 Average CV Score: {np.mean(cv_scores):.4f}")
    print("\n📝 Next Steps:")
    print("  1. Run inference_cv.py to generate submission")
    print("  2. Submit to Dacon")
    print("  3. Expected: 14.5-15.5 points!")