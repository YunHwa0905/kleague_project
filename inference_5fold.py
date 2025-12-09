"""
5-Fold CV 앙상블 추론
- 5개 모델의 예측 평균
- 더 안정적이고 정확한 결과
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

from feature_extractor_v4 import FeatureExtractorV4


def load_cv_models():
    """5-Fold CV 모델 로드"""
    model_dir = Path("D:/workspce/kleague_project/models")
    
    print("Loading 5-Fold CV models...")
    
    # 메타 정보 로드
    meta = joblib.load(model_dir / 'cv_meta.pkl')
    n_folds = meta['n_folds']
    feature_cols = joblib.load(model_dir / 'feature_cols_cv.pkl')
    
    print(f"✅ CV Score: {meta['avg_score']:.4f} ± {meta['std_score']:.4f}")
    print(f"✅ Number of folds: {n_folds}")
    print(f"✅ Features: {len(feature_cols)}")
    
    # 각 Fold 모델 로드
    models_x = []
    models_y = []
    
    for i in range(1, n_folds + 1):
        model_x = lgb.Booster(model_file=str(model_dir / f'lgbm_cv_x_fold{i}.txt'))
        model_y = lgb.Booster(model_file=str(model_dir / f'lgbm_cv_y_fold{i}.txt'))
        models_x.append(model_x)
        models_y.append(model_y)
        print(f"  ✅ Loaded Fold {i} models")
    
    return models_x, models_y, feature_cols


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
    
    extractor = FeatureExtractorV4()
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


def predict_ensemble(models_x, models_y, df_features, feature_cols):
    """5개 모델의 앙상블 예측"""
    print("\n🎯 Making ensemble predictions...")
    
    X_test = df_features[feature_cols]
    n_folds = len(models_x)
    
    # 각 모델의 예측 저장
    all_pred_x = []
    all_pred_y = []
    
    for i, (model_x, model_y) in enumerate(zip(models_x, models_y), 1):
        print(f"  Predicting with Fold {i} model...")
        
        pred_x = model_x.predict(X_test)
        pred_y = model_y.predict(X_test)
        
        all_pred_x.append(pred_x)
        all_pred_y.append(pred_y)
    
    # 앙상블: 평균
    print("\n📊 Computing ensemble (averaging)...")
    
    pred_x_ensemble = np.mean(all_pred_x, axis=0)
    pred_y_ensemble = np.mean(all_pred_y, axis=0)
    
    # 좌표 범위 제한
    pred_x_ensemble = np.clip(pred_x_ensemble, 0, 105)
    pred_y_ensemble = np.clip(pred_y_ensemble, 0, 68)
    
    df_pred = pd.DataFrame({
        'game_episode': df_features['game_episode'],
        'end_x': pred_x_ensemble,
        'end_y': pred_y_ensemble
    })
    
    print(f"✅ Ensemble predictions completed")
    print(f"\nPrediction Statistics:")
    print(f"  X - Mean: {pred_x_ensemble.mean():.2f}, Std: {pred_x_ensemble.std():.2f}")
    print(f"  Y - Mean: {pred_y_ensemble.mean():.2f}, Std: {pred_y_ensemble.std():.2f}")
    
    # 개별 모델 분산도 확인
    print(f"\nModel Agreement:")
    print(f"  X std across models: {np.mean([np.std(x) for x in zip(*all_pred_x)]):.2f}")
    print(f"  Y std across models: {np.mean([np.std(y) for y in zip(*all_pred_y)]):.2f}")
    
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
    filename = f"submission_cv_{timestamp}.csv"
    output_path = output_dir / filename
    
    df_submission.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"\n✅ Submission file created:")
    print(f"   {output_path}")
    print(f"\n📊 Submission Info:")
    print(f"   Total episodes: {len(df_submission)}")
    
    return output_path


def main():
    print("="*70)
    print("🎯 5-Fold CV Ensemble Inference")
    print("="*70)
    
    # 1. 모델 로드
    models_x, models_y, feature_cols = load_cv_models()
    
    # 2. test 데이터 로드
    test_episodes, test_csv = load_test_data()
    
    # 3. Feature 추출
    df_features = extract_test_features(test_episodes)
    
    # 4. 앙상블 예측
    df_pred = predict_ensemble(models_x, models_y, df_features, feature_cols)
    
    # 5. 제출 파일 생성
    output_path = create_submission(df_pred)
    
    print("\n" + "="*70)
    print("✅ Ensemble Inference Completed!")
    print("="*70)
    print(f"\n📝 Next: Submit {output_path.name} to Dacon!")
    print("\n🎯 Expected Public Score: 14.5-15.5")


if __name__ == "__main__":
    main()