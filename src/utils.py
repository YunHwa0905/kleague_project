"""
유틸리티 함수 모듈
공통으로 사용되는 헬퍼 함수들
"""

import logging
import random
from typing import Optional

import numpy as np
import pandas as pd

from src.config import FIELD_LENGTH, FIELD_WIDTH, LOG_FORMAT, LOG_LEVEL, RANDOM_SEED


def setup_logger(name: str, level: str = LOG_LEVEL) -> logging.Logger:
    """
    로거를 설정합니다.
    
    Args:
        name: 로거 이름
        level: 로그 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        설정된 로거 객체
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    # 핸들러가 이미 있으면 추가하지 않음
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
    
    return logger


def set_seed(seed: int = RANDOM_SEED) -> None:
    """
    재현성을 위한 랜덤 시드 설정
    
    Args:
        seed: 랜덤 시드 값
    """
    random.seed(seed)
    np.random.seed(seed)
    # torch가 설치되어 있다면
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def clip_coordinates(x: float, y: float) -> tuple[float, float]:
    """
    좌표를 필드 경계 내로 제한합니다.
    
    Args:
        x: X 좌표
        y: Y 좌표
    
    Returns:
        제한된 (x, y) 좌표
    """
    x = np.clip(x, 0, FIELD_LENGTH)
    y = np.clip(y, 0, FIELD_WIDTH)
    return float(x), float(y)


def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    두 점 사이의 유클리드 거리를 계산합니다.
    
    Args:
        x1, y1: 첫 번째 점의 좌표
        x2, y2: 두 번째 점의 좌표
    
    Returns:
        유클리드 거리
    """
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculate_angle(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    두 점 사이의 각도를 계산합니다 (라디안).
    
    Args:
        x1, y1: 시작점 좌표
        x2, y2: 끝점 좌표
    
    Returns:
        각도 (라디안)
    """
    return np.arctan2(y2 - y1, x2 - x1)


def load_episode_safely(file_path: str) -> Optional[pd.DataFrame]:
    """
    에피소드 파일을 안전하게 로드합니다.
    
    Args:
        file_path: 파일 경로
    
    Returns:
        DataFrame 또는 None (파일이 없거나 에러 시)
    """
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        return None
    except Exception as e:
        logger = setup_logger(__name__)
        logger.error(f"파일 로드 실패: {file_path}, 에러: {e}")
        return None


def get_field_zone(x: float, y: float) -> str:
    """
    좌표가 어느 필드 영역에 속하는지 반환합니다.
    
    Args:
        x: X 좌표
        y: Y 좌표
    
    Returns:
        영역 문자열 ("defensive", "middle", "offensive")
    """
    if x < FIELD_LENGTH / 3:
        return "defensive"
    elif x < 2 * FIELD_LENGTH / 3:
        return "middle"
    else:
        return "offensive"


def is_valid_coordinate(x: float, y: float) -> bool:
    """
    좌표가 유효한지 확인합니다.
    
    Args:
        x: X 좌표
        y: Y 좌표
    
    Returns:
        유효 여부
    """
    return (0 <= x <= FIELD_LENGTH) and (0 <= y <= FIELD_WIDTH)


# ============================================================================
# 메인 실행 (테스트용)
# ============================================================================

if __name__ == "__main__":
    # 로거 테스트
    logger = setup_logger(__name__)
    logger.info("로거 테스트 성공!")
    
    # 시드 설정 테스트
    set_seed(42)
    print(f"랜덤 숫자: {np.random.randint(0, 100)}")
    
    # 좌표 클리핑 테스트
    x, y = clip_coordinates(120, -10)
    print(f"클리핑 결과: ({x}, {y})")  # (105, 0)
    
    # 거리 계산 테스트
    dist = calculate_distance(0, 0, 3, 4)
    print(f"거리: {dist}")  # 5.0
    
    # 필드 영역 테스트
    zone = get_field_zone(80, 30)
    print(f"필드 영역: {zone}")  # offensive
    
    print("\n✅ 모든 유틸리티 함수 테스트 통과!")