"""
Optuna 튜닝된 모델로 추론
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


def load_tuned_models():
    """튜닝된 모델 로드"""
    model_dir = Path("D:/workspce/kleague_project/models")
    
    print("Loading tuned models...")
    
    model_x = lgb.Booster(model_file=str(model_dir / 'lgbm_tuned_x.txt'))
    model_y = lgb.Booster(model_file=str(model_dir / 'lgbm_tuned_y.txt'))
    feature_cols = joblib.load(model_dir / 'feature_cols_tuned.pkl')
    
    print(f"✅ Loaded tuned models")
    print(f"✅ Features: {len(feature_cols)}")
    
    return model_x, model_y, feature_cols


def load_test_data():
    """test 데이터 로드"""
    print("\nLoading test data...")
    
    test_csv = pd.read_csv("D:/workspce/kleague_project/data/test.csv")
    print(f"✅ Test episodes: {len(test_csv)}")
    
    test_episodes = {}
    
    for _, row in tqdm(test_csv.iterrows(), total=len(test_csv), desc="Loading episodes"):
        game_id = row['game_id']
        game_episode = row['game_episode']
        path = row['path']
        
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
            continue
    
    df_features = pd.DataFrame(features_list)
    print(f"✅ Extracted features for {len(df_features)} episodes")
    
    return df_features


def predict(model_x, model_y, df_features, feature_cols):
    """예측 수행"""
    print("\nMaking predictions...")
    
    X_test = df_features[feature_cols]
    
    pred_x = model_x.predict(X_test)
    pred_y = model_y.predict(X_test)
    
    pred_x = np.clip(pred_x, 0, 105)
    pred_y = np.clip(pred_y, 0, 68)
    
    df_pred = pd.DataFrame({
        'game_episode': df_features['game_episode'],
        'end_x': pred_x,
        'end_y': pred_y
    })
    
    print(f"✅ Predictions completed")
    print(f"\nPrediction Statistics:")
    print(f"  X - Mean: {pred_x.mean():.2f}, Std: {pred_x.std():.2f}")
    print(f"  Y - Mean: {pred_y.mean():.2f}, Std: {pred_y.std():.2f}")
    
    return df_pred


def create_submission(df_pred):
    """제출 파일 생성"""
    print("\nCreating submission file...")
    
    sample = pd.read_csv("D:/workspce/kleague_project/data/sample_submission.csv")
    
    df_submission = sample[['game_episode']].merge(
        df_pred,
        on='game_episode',
        how='left'
    )
    
    missing = df_submission['end_x'].isna().sum()
    if missing > 0:
        print(f"⚠️  Warning: {missing} episodes have missing predictions")
        df_submission['end_x'].fillna(52.5, inplace=True)
        df_submission['end_y'].fillna(34, inplace=True)
    
    output_dir = Path("D:/workspce/kleague_project/outputs")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"submission_tuned_{timestamp}.csv"
    output_path = output_dir / filename
    
    df_submission.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"\n✅ Submission file created:")
    print(f"   {output_path}")
    
    return output_path


def main():
    print("="*70)
    print("🎯 Inference with Tuned Model")
    print("="*70)
    
    model_x, model_y, feature_cols = load_tuned_models()
    test_episodes, test_csv = load_test_data()
    df_features = extract_test_features(test_episodes)
    df_pred = predict(model_x, model_y, df_features, feature_cols)
    output_path = create_submission(df_pred)
    
    print("\n" + "="*70)
    print("✅ Inference Completed!")
    print("="*70)
    print(f"\n📝 Next: Submit {output_path.name} to Dacon!")


if __name__ == "__main__":
    main()