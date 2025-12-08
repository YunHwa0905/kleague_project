"""
설정 관리 모듈
모든 경로, 상수, 하이퍼파라미터를 중앙에서 관리
"""

from pathlib import Path
from typing import Final

# ============================================================================
# 경로 설정
# ============================================================================

PROJECT_ROOT: Final[Path] = Path(__file__).parent.parent
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
OUTPUT_DIR: Final[Path] = PROJECT_ROOT / "outputs"
MODEL_DIR: Final[Path] = PROJECT_ROOT / "models"
NOTEBOOK_DIR: Final[Path] = PROJECT_ROOT / "notebooks"

# 데이터 파일 경로
TEST_CSV: Final[Path] = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_CSV: Final[Path] = DATA_DIR / "sample_submission.csv"
MATCH_INFO_CSV: Final[Path] = DATA_DIR / "match_info.csv"

# ============================================================================
# 축구 필드 상수
# ============================================================================

FIELD_LENGTH: Final[int] = 105  # X축 (미터)
FIELD_WIDTH: Final[int] = 68    # Y축 (미터)

# ============================================================================
# 모델 설정
# ============================================================================

RANDOM_SEED: Final[int] = 42

# LightGBM 기본 파라미터
LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "random_state": RANDOM_SEED,
}

# ============================================================================
# 로깅 설정
# ============================================================================

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"


# ============================================================================
# 디렉토리 생성 함수
# ============================================================================

def setup_directories() -> None:
    """필요한 디렉토리들을 생성합니다."""
    directories = [DATA_DIR, OUTPUT_DIR, MODEL_DIR, NOTEBOOK_DIR]
    for directory in directories:
        directory.mkdir(exist_ok=True, parents=True)


def validate_data_files() -> bool:
    """데이터 파일들이 존재하는지 확인합니다."""
    required_files = [TEST_CSV, SAMPLE_SUBMISSION_CSV]
    missing_files = [f for f in required_files if not f.exists()]
    
    if missing_files:
        print("❌ 누락된 파일:")
        for f in missing_files:
            print(f"   - {f}")
        return False
    
    print("✅ 모든 필수 데이터 파일이 존재합니다.")
    return True


# ============================================================================
# 메인 실행 (테스트용)
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("설정 확인")
    print("=" * 80)
    
    setup_directories()
    
    print(f"\n📁 경로 설정:")
    print(f"   프로젝트 루트: {PROJECT_ROOT}")
    print(f"   데이터 디렉토리: {DATA_DIR}")
    print(f"   출력 디렉토리: {OUTPUT_DIR}")
    print(f"   모델 디렉토리: {MODEL_DIR}")
    
    print(f"\n⚽ 필드 설정:")
    print(f"   크기: {FIELD_LENGTH} x {FIELD_WIDTH}")
    
    print(f"\n🔧 모델 설정:")
    print(f"   랜덤 시드: {RANDOM_SEED}")
    
    print(f"\n📄 데이터 파일 확인:")
    validate_data_files()
    
    print("\n" + "=" * 80)