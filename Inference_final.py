"""
최종 추론 스크립트
- test 데이터에서 예측
- 학습된 모델 사용
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
import joblib
from tqdm import tqdm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from feature_extractor_v3 import FeatureExtractorV3


def load_models():
    """학습된 모델 로드"""
    model_dir = Path("D:/workspce/kleague_project/models")
    
    print("Loading trained models...")
    
    model_x = lgb.Booster(model_file=str(model_dir / 'lgbm_final_x.txt'))
    model_y = lgb.Booster(model_file=str(model_dir / 'lgbm_final_y.txt'))
    feature_cols = joblib.load(model_dir / 'feature_cols_final.pkl')
    
    print(f"✅ Loaded models from {model_dir}/")
    print(f"✅ Features: {len(feature_cols)}")
    
    return model_x, model_y, feature_cols


def load_test_data():
    """test 데이터 로드"""
    print("\nLoading test data...")
    
    # test.csv 로드
    test_csv = pd.read_csv("D:/workspce/kleague_project/data/test.csv")
    
    print(f"✅ Test episodes: {len(test_csv)}")
    
    # 각 에피소드 파일 로드
    test_episodes = {}
    
    for _, row in tqdm(test_csv.iterrows(), total=len(test_csv), desc="Loading episodes"):
        game_id = row['game_id']
        game_episode = row['game_episode']
        path = row['path']
        
        # 파일 경로 생성
        file_path = Path("D:/workspce/kleague_project/data") / path
        
        try:
            df = pd.read_csv(file_path)
            test_episodes[game_episode] = df
        except Exception as e:
            print(f"\nWarning: Failed to load {game_episode}: {e}")
            continue
    
    print(f"✅ Loaded {len(test_episodes)} test episodes")
    
    return test_episodes, test_csv


def extract_test_features(test_episodes):
    """test 데이터에서 Feature 추출"""
    print("\nExtracting features from test data...")
    
    extractor = FeatureExtractorV3()
    features_list = []
    
    for game_episode, df_episode in tqdm(test_episodes.items(), desc="Extracting features"):
        try:
            features = extractor.extract_features(df_episode)
            features['game_episode'] = game_episode
            features_list.append(features)
        except Exception as e:
            print(f"\nWarning: Failed to extract features for {game_episode}: {e}")
            # 기본값으로 채우기
            features = {col: 0 for col in extractor.extract_features(df_episode).keys()}
            features['game_episode'] = game_episode
            features_list.append(features)
            continue
    
    df_features = pd.DataFrame(features_list)
    
    print(f"✅ Extracted features for {len(df_features)} episodes")
    
    return df_features


def predict(model_x, model_y, df_features, feature_cols):
    """예측 수행"""
    print("\nMaking predictions...")
    
    # Feature 순서 맞추기
    X_test = df_features[feature_cols]
    
    # 예측
    pred_x = model_x.predict(X_test)
    pred_y = model_y.predict(X_test)
    
    # 좌표 범위 제한 (필드 크기: 105 x 68)
    pred_x = np.clip(pred_x, 0, 105)
    pred_y = np.clip(pred_y, 0, 68)
    
    # 결과 DataFrame 생성
    df_pred = pd.DataFrame({
        'game_episode': df_features['game_episode'],
        'end_x': pred_x,
        'end_y': pred_y
    })
    
    print(f"✅ Predictions completed")
    print(f"\nPrediction Statistics:")
    print(f"  X - Mean: {pred_x.mean():.2f}, Std: {pred_x.std():.2f}, Range: [{pred_x.min():.2f}, {pred_x.max():.2f}]")
    print(f"  Y - Mean: {pred_y.mean():.2f}, Std: {pred_y.std():.2f}, Range: [{pred_y.min():.2f}, {pred_y.max():.2f}]")
    
    return df_pred


def create_submission(df_pred):
    """제출 파일 생성"""
    print("\nCreating submission file...")
    
    # sample_submission.csv 로드
    sample = pd.read_csv("D:/workspce/kleague_project/data/sample_submission.csv")
    
    # 예측 결과 병합
    df_submission = sample[['game_episode']].merge(
        df_pred,
        on='game_episode',
        how='left'
    )
    
    # 누락된 값 체크
    missing = df_submission['end_x'].isna().sum()
    if missing > 0:
        print(f"⚠️  Warning: {missing} episodes have missing predictions")
        print("   Filling with field center: (52.5, 34)")
        df_submission['end_x'].fillna(52.5, inplace=True)
        df_submission['end_y'].fillna(34, inplace=True)
    
    # 저장
    output_dir = Path("D:/workspce/kleague_project/outputs")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"submission_final_{timestamp}.csv"
    output_path = output_dir / filename
    
    df_submission.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"\n✅ Submission file created:")
    print(f"   {output_path}")
    print(f"\n📊 Submission Info:")
    print(f"   Total episodes: {len(df_submission)}")
    print(f"   Sample predictions:")
    print(df_submission.head(10))
    
    return output_path


def main():
    print("="*70)
    print("🎯 Final Inference - Proper Implementation")
    print("="*70)
    
    # 1. 모델 로드
    model_x, model_y, feature_cols = load_models()
    
    # 2. test 데이터 로드
    test_episodes, test_csv = load_test_data()
    
    # 3. Feature 추출
    df_features = extract_test_features(test_episodes)
    
    # 4. 예측
    df_pred = predict(model_x, model_y, df_features, feature_cols)
    
    # 5. 제출 파일 생성
    output_path = create_submission(df_pred)
    
    print("\n" + "="*70)
    print("✅ Inference Completed Successfully!")
    print("="*70)
    print(f"\n📝 Next: Submit {output_path.name} to Dacon!")


if __name__ == "__main__":
    main()