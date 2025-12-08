"""
추론 (Inference) 스크립트 v2
테스트 데이터에 대해 예측하고 제출 파일 생성
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pandas as pd
from tqdm import tqdm
from datetime import datetime

from src.data_loader import DataLoader
from src.models.baseline import StatisticalBaselineV2, ImprovedBaselineV2
from src.config import OUTPUT_DIR, SAMPLE_SUBMISSION_CSV
from src.utils import setup_logger, set_seed

logger = setup_logger(__name__)


def predict_all(model, loader: DataLoader, show_progress: bool = True):
    """
    모든 테스트 에피소드에 대해 예측합니다.
    
    Args:
        model: 예측 모델
        loader: 데이터 로더
        show_progress: 진행 상황 표시 여부
    
    Returns:
        predictions: 예측 결과 DataFrame
    """
    logger.info(f"모델: {model}")
    logger.info("예측 시작...")
    
    # 테스트 정보 로드
    test_info = loader.load_test_info()
    
    predictions = []
    
    iterator = test_info['game_episode'].values
    if show_progress:
        iterator = tqdm(iterator, desc="예측 중")
    
    for game_episode in iterator:
        # 에피소드 로드
        episode_df = loader.load_episode(game_episode)
        
        # 예측
        pred_x, pred_y = model.predict(episode_df)
        
        predictions.append({
            'game_episode': game_episode,
            'end_x': pred_x,
            'end_y': pred_y
        })
    
    pred_df = pd.DataFrame(predictions)
    
    logger.info(f"✓ {len(predictions)}개 예측 완료")
    
    return pred_df


def create_submission(predictions: pd.DataFrame, 
                     output_path: Path,
                     model_name: str = "baseline"):
    """
    제출 파일을 생성합니다.
    
    Args:
        predictions: 예측 결과 DataFrame
        output_path: 저장 경로
        model_name: 모델 이름
    """
    logger.info("제출 파일 생성 중...")
    
    # sample_submission 형식에 맞추기
    submission = pd.read_csv(SAMPLE_SUBMISSION_CSV)
    
    # game_episode 순서 맞추기
    submission = submission[['game_episode']].merge(
        predictions,
        on='game_episode',
        how='left'
    )
    
    # 결측값 확인
    missing = submission['end_x'].isna().sum()
    if missing > 0:
        logger.warning(f"결측값 {missing}개 발견 - 기본값으로 채움")
        submission['end_x'] = submission['end_x'].fillna(57.75)  # 필드 55% 지점
        submission['end_y'] = submission['end_y'].fillna(34.0)   # 필드 중앙
    
    # 저장
    submission.to_csv(output_path, index=False, encoding='utf-8')
    
    logger.info(f"✓ 제출 파일 저장: {output_path}")
    
    # 통계
    logger.info("\n예측 통계:")
    logger.info(f"  end_x - 평균: {submission['end_x'].mean():.2f}, "
               f"범위: [{submission['end_x'].min():.2f}, {submission['end_x'].max():.2f}]")
    logger.info(f"  end_y - 평균: {submission['end_y'].mean():.2f}, "
               f"범위: [{submission['end_y'].min():.2f}, {submission['end_y'].max():.2f}]")
    
    return submission


def main():
    """메인 함수"""
    print("=" * 80)
    print("K리그 패스 좌표 예측 - 추론 v2 (Y축 예측 개선)")
    print("=" * 80)
    
    # 시드 설정
    set_seed(42)
    
    # 데이터 로더
    loader = DataLoader()
    
    # 모델 선택
    print("\n사용할 모델을 선택하세요:")
    print("1. StatisticalBaseline v2 (기본 + Y축 예측)")
    print("2. ImprovedBaseline v2 (전진 바이어스 + Y축 예측) - 추천!")
    
    choice = input("\n선택 (1 또는 2, Enter는 2번 기본값): ").strip()
    
    if choice == "1":
        model = StatisticalBaselineV2(recent_weight=0.7, use_successful_only=True)
        model_name = "statistical_baseline_v2"
    else:
        model = ImprovedBaselineV2(recent_weight=0.7, forward_bias=0.3, use_successful_only=True)
        model_name = "improved_baseline_v2"
    
    print(f"\n선택된 모델: {model}")
    
    # 예측
    predictions = predict_all(model, loader, show_progress=True)
    
    # 제출 파일 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"submission_{model_name}_{timestamp}.csv"
    
    submission = create_submission(predictions, output_path, model_name)
    
    print("\n" + "=" * 80)
    print("✓ 완료!")
    print("=" * 80)
    print(f"\n제출 파일: {output_path}")
    print(f"\n📊 예측 미리보기:")
    print(submission.head(10).to_string(index=False))
    print(f"\n다음 단계:")
    print(f"  1. {output_path.name} 파일 확인")
    print(f"  2. Dacon 대회 페이지에서 제출")
    print(f"  3. 점수 확인! (v1: 35.99 → v2: 예상 20-25)")
    print("=" * 80)


if __name__ == "__main__":
    main()