"""
LightGBM + XGBoost 앙상블 추론
- 두 모델의 예측 평균
- 최고 성능 기대!
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
import xgboost as xgb
import joblib
from tqdm import tqdm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from feature_extractor_v4 import FeatureExtractorV4


def load_ensemble_models():
    """LightGBM + XGBoost 모델 로드"""
    model_dir = Path("D:/workspce/kleague_project/models")
    
    print("Loading ensemble models (LightGBM + XGBoost)...")
    
    # LightGBM 메타 정보
    lgb_meta = joblib.load(model_dir / 'cv_meta.pkl')
    lgb_feature_cols = joblib.load(model_dir / 'feature_cols_cv.pkl')
    
    print(f"\n📊 LightGBM:")
    print(f"  CV Score: {lgb_meta['avg_score']:.4f} ± {lgb_meta['std_score']:.4f}")
    print(f"  Folds: {lgb_meta['n_folds']}")
    
    # XGBoost 메타 정보
    xgb_meta = joblib.load(model_dir / 'xgb_meta.pkl')
    xgb_feature_cols = joblib.load(model_dir / 'feature_cols_xgb.pkl')
    
    print(f"\n📊 XGBoost:")
    print(f"  CV Score: {xgb_meta['avg_score']:.4f} ± {xgb_meta['std_score']:.4f}")
    print(f"  Folds: {xgb_meta['n_folds']}")
    
    n_folds = lgb_meta['n_folds']
    
    # LightGBM 모델 로드
    print(f"\n⚡ Loading LightGBM models...")
    lgb_models_x = []
    lgb_models_y = []
    
    for i in range(1, n_folds + 1):
        model_x = lgb.Booster(model_file=str(model_dir / f'lgbm_cv_x_fold{i}.txt'))
        model_y = lgb.Booster(model_file=str(model_dir / f'lgbm_cv_y_fold{i}.txt'))
        lgb_models_x.append(model_x)
        lgb_models_y.append(model_y)
    
    print(f"  ✅ Loaded {n_folds} LightGBM models")
    
    # XGBoost 모델 로드
    print(f"\n🚀 Loading XGBoost models...")
    xgb_models_x = []
    xgb_models_y = []
    
    for i in range(1, n_folds + 1):
        model_x = xgb.Booster()
        model_x.load_model(str(model_dir / f'xgb_cv_x_fold{i}.json'))
        model_y = xgb.Booster()
        model_y.load_model(str(model_dir / f'xgb_cv_y_fold{i}.json'))
        xgb_models_x.append(model_x)
        xgb_models_y.append(model_y)
    
    print(f"  ✅ Loaded {n_folds} XGBoost models")
    
    return lgb_models_x, lgb_models_y, xgb_models_x, xgb_models_y, lgb_feature_cols


def load_test_data():
    """test 데이터 로드"""
    print("\nLoading test data...")
    
    test_csv = pd.read_csv("D:/workspce/kleague_project/data/test.csv")
    print(f"✅ Test episodes: {len(test_csv)}")
    
    test_episodes = {}
    
    for _, row in tqdm(test_csv.iterrows(), total=len(test_csv), desc="Loading episodes"):
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


def predict_ensemble(lgb_models_x, lgb_models_y, xgb_models_x, xgb_models_y, 
                     df_features, feature_cols, lgb_weight=0.5):
    """LightGBM + XGBoost 앙상블 예측"""
    print("\n🎯 Making ensemble predictions (LightGBM + XGBoost)...")
    print(f"   Weight: LightGBM {lgb_weight:.1f} / XGBoost {1-lgb_weight:.1f}")
    
    X_test = df_features[feature_cols]
    n_folds = len(lgb_models_x)
    
    # LightGBM 예측
    print("\n⚡ Predicting with LightGBM...")
    lgb_pred_x_all = []
    lgb_pred_y_all = []
    
    for i, (model_x, model_y) in enumerate(zip(lgb_models_x, lgb_models_y), 1):
        pred_x = model_x.predict(X_test)
        pred_y = model_y.predict(X_test)
        lgb_pred_x_all.append(pred_x)
        lgb_pred_y_all.append(pred_y)
    
    lgb_pred_x = np.mean(lgb_pred_x_all, axis=0)
    lgb_pred_y = np.mean(lgb_pred_y_all, axis=0)
    
    print(f"  ✅ LightGBM ensemble done")
    
    # XGBoost 예측
    print("\n🚀 Predicting with XGBoost...")
    xgb_pred_x_all = []
    xgb_pred_y_all = []
    
    dtest = xgb.DMatrix(X_test)
    
    for i, (model_x, model_y) in enumerate(zip(xgb_models_x, xgb_models_y), 1):
        pred_x = model_x.predict(dtest)
        pred_y = model_y.predict(dtest)
        xgb_pred_x_all.append(pred_x)
        xgb_pred_y_all.append(pred_y)
    
    xgb_pred_x = np.mean(xgb_pred_x_all, axis=0)
    xgb_pred_y = np.mean(xgb_pred_y_all, axis=0)
    
    print(f"  ✅ XGBoost ensemble done")
    
    # 최종 앙상블 (가중 평균)
    print(f"\n🎨 Computing weighted ensemble...")
    
    final_pred_x = lgb_pred_x * lgb_weight + xgb_pred_x * (1 - lgb_weight)
    final_pred_y = lgb_pred_y * lgb_weight + xgb_pred_y * (1 - lgb_weight)
    
    # 좌표 범위 제한
    final_pred_x = np.clip(final_pred_x, 0, 105)
    final_pred_y = np.clip(final_pred_y, 0, 68)
    
    df_pred = pd.DataFrame({
        'game_episode': df_features['game_episode'],
        'end_x': final_pred_x,
        'end_y': final_pred_y
    })
    
    print(f"✅ Ensemble predictions completed")
    print(f"\nPrediction Statistics:")
    print(f"  X - Mean: {final_pred_x.mean():.2f}, Std: {final_pred_x.std():.2f}")
    print(f"  Y - Mean: {final_pred_y.mean():.2f}, Std: {final_pred_y.std():.2f}")
    
    # 모델 간 차이 분석
    diff_x = np.abs(lgb_pred_x - xgb_pred_x)
    diff_y = np.abs(lgb_pred_y - xgb_pred_y)
    print(f"\nModel Agreement:")
    print(f"  X difference: {diff_x.mean():.2f} ± {diff_x.std():.2f}")
    print(f"  Y difference: {diff_y.mean():.2f} ± {diff_y.std():.2f}")
    
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
    filename = f"submission_ensemble_{timestamp}.csv"
    output_path = output_dir / filename
    
    df_submission.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"\n✅ Submission file created:")
    print(f"   {output_path}")
    print(f"\n📊 Submission Info:")
    print(f"   Total episodes: {len(df_submission)}")
    
    return output_path


def main():
    print("="*70)
    print("🎯 LightGBM + XGBoost Ensemble Inference")
    print("="*70)
    
    # 1. 모델 로드
    lgb_models_x, lgb_models_y, xgb_models_x, xgb_models_y, feature_cols = load_ensemble_models()
    
    # 2. test 데이터 로드
    test_episodes, test_csv = load_test_data()
    
    # 3. Feature 추출
    df_features = extract_test_features(test_episodes)
    
    # 4. 앙상블 예측 (가중치 0.5:0.5)
    df_pred = predict_ensemble(
        lgb_models_x, lgb_models_y, 
        xgb_models_x, xgb_models_y, 
        df_features, feature_cols, 
        lgb_weight=0.5  # 동일 가중치
    )
    
    # 5. 제출 파일 생성
    output_path = create_submission(df_pred)
    
    print("\n" + "="*70)
    print("✅ Ensemble Inference Completed!")
    print("="*70)
    print(f"\n📝 Next: Submit {output_path.name} to Dacon!")
    print("\n🎯 Expected Public Score: 14.8-15.2")


if __name__ == "__main__":
    main()