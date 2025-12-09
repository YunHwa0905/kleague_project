"""
최종 올바른 LightGBM 학습 스크립트
- train.csv에서 정답 포함 데이터 로드
- game_id 기반 train/val split (Data Leakage 방지)
- Episode 독립성 보장
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
    """
    train.csv 로드
    
    주의: 사용자 로컬 환경에서 실행 시 경로 수정 필요
    D:/workspce/kleague_project/data/train.csv
    """
    # 로컬 경로 (사용자 환경에 맞게 수정)
    train_path = "D:/workspce/kleague_project/data/train.csv"
    
    print("Loading train.csv...")
    print(f"Path: {train_path}")
    
    df_train = pd.read_csv(train_path)
    
    print(f"✅ Loaded {len(df_train):,} rows")
    print(f"✅ Unique episodes: {df_train['game_episode'].nunique():,}")
    print(f"✅ Unique games: {df_train['game_id'].nunique():,}")
    
    return df_train


def prepare_training_data(df_train):
    """
    학습 데이터 준비
    
    핵심:
    1. 각 에피소드별로 그룹화
    2. 마지막 패스의 end_x, end_y를 정답으로 추출
    3. 마지막 패스를 제외한 시퀀스로 Feature 추출
    """
    print("\nPreparing training data...")
    
    extractor = FeatureExtractorV3()
    
    features_list = []
    labels_list = []
    
    # 에피소드별로 처리
    grouped = df_train.groupby('game_episode')
    
    for game_episode, df_episode in tqdm(grouped, desc="Processing episodes"):
        try:
            # 1. 정답 추출: 마지막 패스의 end 좌표
            # 마지막 액션 찾기 (가장 큰 action_id)
            last_action = df_episode.loc[df_episode['action_id'].idxmax()]
            
            # 마지막 패스인지 확인
            if last_action['type_name'] != 'Pass':
                # 마지막이 패스가 아니면, 마지막 패스 찾기
                passes = df_episode[df_episode['type_name'] == 'Pass']
                if len(passes) == 0:
                    continue
                last_pass = passes.iloc[-1]
            else:
                last_pass = last_action
            
            # 정답 좌표
            target_x = last_pass['end_x']
            target_y = last_pass['end_y']
            
            # NaN 체크
            if pd.isna(target_x) or pd.isna(target_y):
                continue
            
            # 2. Feature 추출
            # 마지막 패스를 제외한 데이터 생성
            df_for_features = df_episode[df_episode['action_id'] < last_pass['action_id']].copy()
            
            # 마지막 패스의 start 정보를 추가 (Feature 추출에 필요)
            last_pass_info = last_pass.copy()
            last_pass_info['end_x'] = np.nan  # Feature 추출 시 end는 사용 안 함
            last_pass_info['end_y'] = np.nan
            
            df_with_last_start = pd.concat([df_for_features, last_pass_info.to_frame().T], ignore_index=True)
            
            # Feature 추출
            features = extractor.extract_features(df_with_last_start)
            
            # 저장
            features['game_episode'] = game_episode
            features['game_id'] = last_pass['game_id']
            features_list.append(features)
            
            labels_list.append({
                'game_episode': game_episode,
                'target_x': target_x,
                'target_y': target_y
            })
            
        except Exception as e:
            print(f"\nWarning: Failed to process {game_episode}: {e}")
            continue
    
    # DataFrame 생성
    df_features = pd.DataFrame(features_list)
    df_labels = pd.DataFrame(labels_list)
    
    # 병합
    df_final = df_features.merge(df_labels, on='game_episode')
    
    print(f"\n✅ Prepared {len(df_final):,} training samples")
    print(f"✅ Features: {len([c for c in df_final.columns if c not in ['game_episode', 'game_id', 'target_x', 'target_y']])}")
    
    return df_final


def train_with_game_split(df_train, test_size=0.2, random_state=42):
    """
    game_id 기반으로 train/val split
    
    같은 게임의 에피소드는 train 또는 val에만 존재
    → Data Leakage 완전 방지!
    """
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
    
    # LightGBM 파라미터 (최적화된 값)
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
    # 로컬 경로
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
    print("🚀 Proper LightGBM Training - Final Version")
    print("="*70)
    
    # 1. 데이터 로드
    df_train_raw = load_train_csv()
    
    # 2. 데이터 준비
    df_train = prepare_training_data(df_train_raw)
    
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
    print("  3. Compare with baseline (27.63)")