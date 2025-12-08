"""
베이스라인 모델 v2: Y축 예측 개선
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional

from src.config import FIELD_LENGTH, FIELD_WIDTH
from src.utils import clip_coordinates, setup_logger

logger = setup_logger(__name__)


class StatisticalBaselineV2:
    """
    개선된 통계 기반 베이스라인 모델 v2
    
    개선 사항:
    1. Y축 예측 추가 (이전에는 34.0 고정)
    2. 패스가 없을 때 더 나은 기본값
    3. 성공한 패스만 사용하는 옵션
    """
    
    def __init__(self, recent_weight: float = 0.7, use_successful_only: bool = True):
        """
        Args:
            recent_weight: 최근 패스에 부여할 가중치 (0~1)
            use_successful_only: 성공한 패스만 사용할지 여부
        """
        self.recent_weight = recent_weight
        self.use_successful_only = use_successful_only
        self.name = "StatisticalBaselineV2"
    
    def predict(self, episode_df: pd.DataFrame) -> Tuple[float, float]:
        """
        에피소드의 다음 패스 위치를 예측합니다.
        
        Args:
            episode_df: 에피소드 데이터
        
        Returns:
            (pred_x, pred_y): 예측 좌표
        """
        # 빈 에피소드
        if episode_df is None or len(episode_df) == 0:
            return self._default_prediction()
        
        # Pass 필터링
        passes = episode_df[episode_df['type_name'] == 'Pass'].copy()
        
        # 성공한 패스만 사용
        if self.use_successful_only and len(passes) > 0:
            passes = passes[passes['result_name'] == 'Successful']
        
        if len(passes) == 0:
            # Pass가 없으면 마지막 액션의 end 위치
            return self._predict_from_last_action(episode_df)
        
        # 마지막 패스의 end 위치
        last_pass = passes.iloc[-1]
        curr_x = last_pass['end_x']
        curr_y = last_pass['end_y']
        
        # NaN 체크 - 더 철저하게
        if pd.isna(curr_x) or pd.isna(curr_y):
            # 마지막 패스가 NaN이면 이전 패스들의 평균 end 위치
            valid_passes = passes.dropna(subset=['end_x', 'end_y'])
            if len(valid_passes) > 0:
                curr_x = valid_passes.iloc[-1]['end_x']
                curr_y = valid_passes.iloc[-1]['end_y']
            else:
                return self._predict_from_last_action(episode_df)
        
        if pd.isna(curr_x) or pd.isna(curr_y):
            return self._default_prediction()
        
        # 평균 이동 벡터 계산
        avg_dx, avg_dy = self._calculate_average_movement(passes)
        
        # 예측
        pred_x = curr_x + avg_dx
        pred_y = curr_y + avg_dy
        
        # 필드 경계 제약
        pred_x, pred_y = clip_coordinates(pred_x, pred_y)
        
        return pred_x, pred_y
    
    def _calculate_average_movement(self, passes: pd.DataFrame) -> Tuple[float, float]:
        """
        패스들의 평균 이동 벡터를 계산합니다.
        
        Args:
            passes: 패스 DataFrame
        
        Returns:
            (avg_dx, avg_dy): 평균 이동 벡터
        """
        if len(passes) == 0:
            return 0.0, 0.0
        
        # NaN 제거
        valid_passes = passes.dropna(subset=['start_x', 'start_y', 'end_x', 'end_y'])
        
        if len(valid_passes) == 0:
            return 0.0, 0.0
        
        # 이동 벡터 계산
        dx_list = []
        dy_list = []
        weights = []
        
        for idx in range(len(valid_passes)):
            row = valid_passes.iloc[idx]
            dx = row['end_x'] - row['start_x']
            dy = row['end_y'] - row['start_y']
            
            dx_list.append(dx)
            dy_list.append(dy)
            
            # 최근 패스에 더 높은 가중치
            if self.recent_weight > 0:
                weight = 1.0 + (idx / len(valid_passes)) * self.recent_weight
            else:
                weight = 1.0
            weights.append(weight)
        
        if len(dx_list) == 0:
            return 0.0, 0.0
        
        # 가중 평균
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        avg_dx = np.average(dx_list, weights=weights)
        avg_dy = np.average(dy_list, weights=weights)
        
        return float(avg_dx), float(avg_dy)
    
    def _predict_from_last_action(self, episode_df: pd.DataFrame) -> Tuple[float, float]:
        """
        패스가 없을 때 마지막 액션의 end 위치 반환
        
        Args:
            episode_df: 에피소드 DataFrame
        
        Returns:
            (x, y): 좌표
        """
        if len(episode_df) == 0:
            return self._default_prediction()
        
        # 뒤에서부터 유효한 좌표 찾기
        for idx in range(len(episode_df) - 1, -1, -1):
            row = episode_df.iloc[idx]
            if pd.notna(row['end_x']) and pd.notna(row['end_y']):
                x, y = clip_coordinates(row['end_x'], row['end_y'])
                return x, y
        
        return self._default_prediction()
    
    def _default_prediction(self) -> Tuple[float, float]:
        """
        기본 예측값 (필드 중앙보다는 약간 공격적)
        
        Returns:
            (x, y): 기본 좌표
        """
        # 필드 중앙보다 약간 앞쪽
        return FIELD_LENGTH * 0.55, FIELD_WIDTH / 2
    
    def __repr__(self) -> str:
        return f"{self.name}(recent_weight={self.recent_weight}, use_successful_only={self.use_successful_only})"


class ImprovedBaselineV2(StatisticalBaselineV2):
    """
    개선된 베이스라인 v2: 공격 방향 + Y축 변화 고려
    
    추가 전략:
    - X축으로 전진하는 경향 반영
    - Y축 변화도 고려
    - 필드 영역별 다른 전략
    """
    
    def __init__(self, 
                 recent_weight: float = 0.7, 
                 forward_bias: float = 0.3,
                 use_successful_only: bool = True):
        """
        Args:
            recent_weight: 최근 패스 가중치
            forward_bias: 전진 바이어스 (0~1)
            use_successful_only: 성공한 패스만 사용
        """
        super().__init__(recent_weight, use_successful_only)
        self.forward_bias = forward_bias
        self.name = "ImprovedBaselineV2"
    
    def predict(self, episode_df: pd.DataFrame) -> Tuple[float, float]:
        """예측 (전진 바이어스 + Y축 변화 포함)"""
        pred_x, pred_y = super().predict(episode_df)
        
        if episode_df is None or len(episode_df) == 0:
            return pred_x, pred_y
        
        # 공격 방향으로 전진 바이어스 적용
        if self.forward_bias > 0:
            passes = episode_df[episode_df['type_name'] == 'Pass']
            
            if self.use_successful_only and len(passes) > 0:
                passes = passes[passes['result_name'] == 'Successful']
            
            valid_passes = passes.dropna(subset=['start_x', 'end_x'])
            
            if len(valid_passes) > 0:
                # 평균 X 진행 방향
                x_progress = (valid_passes['end_x'] - valid_passes['start_x']).mean()
                
                if x_progress > 0:  # 앞으로 가는 경향
                    pred_x += x_progress * self.forward_bias
                    pred_x = np.clip(pred_x, 0, FIELD_LENGTH)
        
        return pred_x, pred_y
    
    def __repr__(self) -> str:
        return f"{self.name}(recent_weight={self.recent_weight}, forward_bias={self.forward_bias})"


# ============================================================================
# 메인 실행 (테스트용)
# ============================================================================

if __name__ == "__main__":
    from src.data_loader import DataLoader
    
    print("=" * 80)
    print("베이스라인 모델 v2 테스트 (Y축 예측 개선)")
    print("=" * 80)
    
    # 데이터 로더
    loader = DataLoader()
    test_info = loader.load_test_info()
    
    # 여러 에피소드 테스트
    print("\n샘플 예측 결과:\n")
    
    for i in range(5):
        game_episode = test_info.iloc[i]['game_episode']
        episode_df = loader.load_episode(game_episode)
        
        # 모델 예측
        model = ImprovedBaselineV2(recent_weight=0.7, forward_bias=0.3)
        pred_x, pred_y = model.predict(episode_df)
        
        print(f"{game_episode}: ({pred_x:.2f}, {pred_y:.2f})")
        
        # 실제 패스 정보
        if episode_df is not None and len(episode_df) > 0:
            passes = episode_df[episode_df['type_name'] == 'Pass']
            if len(passes) > 0:
                last_pass = passes.iloc[-1]
                if pd.notna(last_pass['end_x']) and pd.notna(last_pass['end_y']):
                    print(f"  (마지막 패스: {last_pass['end_x']:.2f}, {last_pass['end_y']:.2f})")
        print()
    
    print("=" * 80)
    print("✓ 테스트 완료!")
    print("\nY축 값이 34.0이 아닌 다양한 값으로 나오는지 확인하세요!")
    print("=" * 80)