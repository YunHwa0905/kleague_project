"""
수정된 LightGBM 학습 스크립트 v2
- 더 많은 에피소드 활용
- 마지막 패스 찾기 로직 개선
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from feature_extractor_v3 import FeatureExtractorV3


def load_train_csv():
    """train.csv 로드"""
    train_path = "D:/workspce/kleague_project/data/train.csv"
    
    print("Loading train.csv...")
    print(f"Path: {train_path}")
    
    df_train = pd.read_csv(train_path)
    
    print(f"✅ Loaded {len(df_train):,} rows")
    print(f"✅ Unique episodes: {df_train['game_episode'].nunique():,}")
    print(f"✅ Unique games: {df_train['game_id'].nunique():,}")
    
    return df_train


def prepare_training_data_v2(df_train):
    """
    개선된 학습 데이터 준비
    
    핵심 개선:
    1. 모든 에피소드의 마지막 액션을 타겟으로
    2. 더 유연한 마지막 패스 찾기
    """
    print("\nPreparing training data (v2 - Improved)...")
    
    extractor = FeatureExtractorV3()
    
    features_list = []
    labels_list = []
    failed_count = 0
    
    # 에피소드별로 처리
    grouped = df_train.groupby('game_episode')
    total_episodes = len(grouped)
    
    print(f"Total episodes to process: {total_episodes:,}\n")
    
    for game_episode, df_episode in tqdm(grouped, desc="Processing episodes"):
        try:
            # 정렬 (action_id 순서대로)
            df_episode = df_episode.sort_values('action_id').reset_index(drop=True)
            
            # 1. 타겟 찾기: 여러 방법 시도
            target_x = None
            target_y = None
            last_pass_idx = None
            
            # 방법 1: type_name이 'Pass'인 마지막 액션
            pass_actions = df_episode[df_episode['type_name'] == 'Pass']
            
            if len(pass_actions) > 0:
                last_pass = pass_actions.iloc[-1]
                target_x = last_pass['end_x']
                target_y = last_pass['end_y']
                last_pass_idx = last_pass.name
                
                # end 좌표가 없으면 다음 방법 시도
                if pd.isna(target_x) or pd.isna(target_y):
                    # 방법 2: 마지막 액션이 패스가 아니면, 그 이전 패스
                    if len(pass_actions) >= 2:
                        last_pass = pass_actions.iloc[-2]
                        target_x = last_pass['end_x']
                        target_y = last_pass['end_y']
                        last_pass_idx = last_pass.name
            
            # 타겟이 여전히 없으면
            if pd.isna(target_x) or pd.isna(target_y):
                # 방법 3: end_x, end_y가 있는 마지막 액션
                valid_actions = df_episode.dropna(subset=['end_x', 'end_y'])
                if len(valid_actions) > 0:
                    last_valid = valid_actions.iloc[-1]
                    target_x = last_valid['end_x']
                    target_y = last_valid['end_y']
                    last_pass_idx = last_valid.name
            
            # 여전히 타겟이 없으면 skip
            if pd.isna(target_x) or pd.isna(target_y):
                failed_count += 1
                continue
            
            # 2. Feature 추출을 위한 데이터 준비
            # 타겟이 된 액션의 start 정보를 마지막 패스 정보로 사용
            target_action = df_episode.loc[last_pass_idx].copy()
            
            # 타겟 액션 이전의 데이터만 사용
            df_for_features = df_episode[df_episode.index < last_pass_idx].copy()
            
            # 마지막 패스의 start 정보 추가 (end는 NaN으로)
            last_pass_row = target_action.copy()
            last_pass_row['end_x'] = np.nan
            last_pass_row['end_y'] = np.nan
            
            df_with_last = pd.concat([df_for_features, last_pass_row.to_frame().T], ignore_index=True)
            
            # 3. Feature 추출
            features = extractor.extract_features(df_with_last)
            
            # 4. 저장
            features['game_episode'] = game_episode
            features['game_id'] = target_action['game_id']
            features_list.append(features)
            
            labels_list.append({
                'game_episode': game_episode,
                'target_x': target_x,
                'target_y': target_y
            })
            
        except Exception as e:
            failed_count += 1
            if failed_count <= 10:  # 처음 10개만 출력
                print(f"\nWarning: Failed to process {game_episode}: {e}")
            continue
    
    # DataFrame 생성
    df_features = pd.DataFrame(features_list)
    df_labels = pd.DataFrame(labels_list)
    
    # 병합
    df_final = df_features.merge(df_labels, on='game_episode')
    
    print(f"\n" + "="*70)
    print(f"✅ Successfully processed: {len(df_final):,} episodes")
    print(f"❌ Failed to process: {failed_count:,} episodes")
    print(f"📊 Success rate: {len(df_final) / total_episodes * 100:.1f}%")
    print(f"✅ Features: {len([c for c in df_final.columns if c not in ['game_episode', 'game_id', 'target_x', 'target_y']])}")
    print("="*70)
    
    return df_final


def train_with_game_split(df_train, test_size=0.2, random_state=42):
    """game_id 기반으로 train/val split"""
    print("\n" + "="*70)
    print("Training with Game-based Split (Data Leakage Free!)")
    print("="*70)
    
    # Feature 컬럼
    feature_cols = [col for col in df_train.columns 
                   if col not in ['game_episode', 'target_x', 'target_y', 'game_id']]
    
    print(f"\n📊 Using {len(feature_cols)} features:")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {col}")
    
    # game_id 기반 split
    print(f"\n🔀 Splitting data by game_id...")
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(gss.split(df_train, groups=df_train['game_id']))
    
    df_train_split = df_train.iloc[train_idx]
    df_val_split = df_train.iloc[val_idx]
    
    train_games = df_train_split['game_id'].nunique()
    val_games = df_val_split['game_id'].nunique()
    
    print(f"\n📈 Train: {len(df_train_split):,} episodes from {train_games} games")
    print(f"📉 Val:   {len(df_val_split):,} episodes from {val_games} games")
    
    # 데이터 준비
    X_train = df_train_split[feature_cols]
    y_train_x = df_train_split['target_x']
    y_train_y = df_train_split['target_y']
    
    X_val = df_val_split[feature_cols]
    y_val_x = df_val_split['target_x']
    y_val_y = df_val_split['target_y']
    
    # LightGBM 파라미터
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 64,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'verbose': -1,
        'random_state': random_state
    }
    
    print(f"\n🎯 LightGBM Parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    
    # X 좌표 모델 학습
    print("\n" + "-"*70)
    print("🔵 Training X coordinate model...")
    print("-"*70)
    
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
    
    # Y 좌표 모델 학습
    print("\n" + "-"*70)
    print("🔴 Training Y coordinate model...")
    print("-"*70)
    
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
    print("\n" + "="*70)
    print("📊 Validation Results")
    print("="*70)
    
    pred_x = model_x.predict(X_val)
    pred_y = model_y.predict(X_val)
    
    # 좌표 범위 제한
    pred_x = np.clip(pred_x, 0, 105)
    pred_y = np.clip(pred_y, 0, 68)
    
    # 유클리드 거리 계산
    distances = np.sqrt((pred_x - y_val_x)**2 + (pred_y - y_val_y)**2)
    
    print(f"\n🎯 Validation Metrics:")
    print(f"  X RMSE: {np.sqrt(np.mean((pred_x - y_val_x)**2)):.4f}")
    print(f"  Y RMSE: {np.sqrt(np.mean((pred_y - y_val_y)**2)):.4f}")
    print(f"  Average Euclidean Distance: {distances.mean():.4f}")
    print(f"  Median Distance: {np.median(distances):.4f}")
    print(f"  Std Distance: {distances.std():.4f}")
    print(f"  Min Distance: {distances.min():.4f}")
    print(f"  Max Distance: {distances.max():.4f}")
    
    print("\n" + "="*70)
    
    return model_x, model_y, feature_cols, distances.mean()


def save_models(model_x, model_y, feature_cols, val_score):
    """모델 저장"""
    output_dir = Path("D:/workspce/kleague_project/models")
    output_dir.mkdir(exist_ok=True)
    
    model_x.save_model(str(output_dir / 'lgbm_final_x.txt'))
    model_y.save_model(str(output_dir / 'lgbm_final_y.txt'))
    joblib.dump(feature_cols, output_dir / 'feature_cols_final.pkl')
    
    # 메타 정보 저장
    meta = {
        'val_score': val_score,
        'num_features': len(feature_cols),
        'feature_names': feature_cols
    }
    joblib.dump(meta, output_dir / 'model_meta.pkl')
    
    print(f"\n💾 Models saved to: {output_dir}/")
    print(f"  - lgbm_final_x.txt")
    print(f"  - lgbm_final_y.txt")
    print(f"  - feature_cols_final.pkl")
    print(f"  - model_meta.pkl")


if __name__ == "__main__":
    print("="*70)
    print("🚀 LightGBM Training v2 - Using More Data!")
    print("="*70)
    
    # 1. 데이터 로드
    df_train_raw = load_train_csv()
    
    # 2. 데이터 준비 (개선된 버전)
    df_train = prepare_training_data_v2(df_train_raw)
    
    # 3. 학습
    model_x, model_y, feature_cols, val_score = train_with_game_split(df_train)
    
    # 4. 저장
    save_models(model_x, model_y, feature_cols, val_score)
    
    print("\n" + "="*70)
    print("✅ Training Completed Successfully!")
    print("="*70)
    print(f"\n🎯 Final Validation Score: {val_score:.4f}")
    print("\n📝 Next Steps:")
    print("  1. Run inference_final.py to generate submission")
    print("  2. Submit to Dacon")
    print("  3. Compare with previous (13.02)")