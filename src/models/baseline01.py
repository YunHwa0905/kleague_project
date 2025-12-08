"""
베이스라인 모델: 단순 통계 기반 예측
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional

from src.config import FIELD_LENGTH, FIELD_WIDTH
from src.utils import clip_coordinates, setup_logger

logger = setup_logger(__name__)


class StatisticalBaseline:
    """
    통계 기반 베이스라인 모델
    
    전략:
    1. 에피소드의 모든 패스를 분석
    2. 각 패스의 이동 벡터 (dx, dy) 계산
    3. 평균 이동 벡터를 구함
    4. 마지막 패스 end 위치에서 평균 벡터만큼 이동
    5. 필드 경계 내로 제한
    """
    
    def __init__(self, recent_weight: float = 0.7):
        """
        Args:
            recent_weight: 최근 패스에 부여할 가중치 (0~1)
                          1.0이면 최근 패스만, 0.0이면 모든 패스 동일 가중치
        """
        self.recent_weight = recent_weight
        self.name = "StatisticalBaseline"
    
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
        
        # Pass만 필터링
        passes = episode_df[episode_df['type_name'] == 'Pass'].copy()
        
        if len(passes) == 0:
            # Pass가 없으면 마지막 액션의 end 위치
            return self._predict_from_last_action(episode_df)
        
        # 마지막 패스의 end 위치
        last_pass = passes.iloc[-1]
        curr_x = last_pass['end_x']
        curr_y = last_pass['end_y']
        
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
        
        # 이동 벡터 계산
        dx_list = []
        dy_list = []
        weights = []
        
        for idx in range(len(passes)):
            row = passes.iloc[idx]
            dx = row['end_x'] - row['start_x']
            dy = row['end_y'] - row['start_y']
            
            if pd.notna(dx) and pd.notna(dy):
                dx_list.append(dx)
                dy_list.append(dy)
                
                # 최근 패스에 더 높은 가중치
                if self.recent_weight > 0:
                    weight = 1.0 + (idx / len(passes)) * self.recent_weight
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
        
        last = episode_df.iloc[-1]
        
        if pd.notna(last['end_x']) and pd.notna(last['end_y']):
            x, y = clip_coordinates(last['end_x'], last['end_y'])
            return x, y
        
        return self._default_prediction()
    
    def _default_prediction(self) -> Tuple[float, float]:
        """
        기본 예측값 (필드 중앙)
        
        Returns:
            (x, y): 필드 중앙 좌표
        """
        return FIELD_LENGTH / 2, FIELD_WIDTH / 2
    
    def __repr__(self) -> str:
        return f"{self.name}(recent_weight={self.recent_weight})"


class ImprovedBaseline(StatisticalBaseline):
    """
    개선된 베이스라인: 공격 방향을 고려
    
    추가 전략:
    - X축으로 전진하는 경향 반영
    - 공격 방향 가중치 추가
    """
    
    def __init__(self, recent_weight: float = 0.7, forward_bias: float = 0.2):
        """
        Args:
            recent_weight: 최근 패스 가중치
            forward_bias: 전진 바이어스 (0~1)
        """
        super().__init__(recent_weight)
        self.forward_bias = forward_bias
        self.name = "ImprovedBaseline"
    
    def predict(self, episode_df: pd.DataFrame) -> Tuple[float, float]:
        """예측 (전진 바이어스 포함)"""
        pred_x, pred_y = super().predict(episode_df)
        
        # 공격 방향으로 약간 더 전진
        if self.forward_bias > 0:
            passes = episode_df[episode_df['type_name'] == 'Pass'] if episode_df is not None and len(episode_df) > 0 else pd.DataFrame()
            
            if len(passes) > 0:
                # 평균 X 진행 방향
                x_progress = (passes['end_x'] - passes['start_x']).mean()
                
                if x_progress > 0:  # 앞으로 가는 경향
                    pred_x += x_progress * self.forward_bias
                    pred_x = np.clip(pred_x, 0, FIELD_LENGTH)
        
        return pred_x, pred_y


# ============================================================================
# 메인 실행 (테스트용)
# ============================================================================

if __name__ == "__main__":
    from src.data_loader import DataLoader
    
    print("=" * 80)
    print("베이스라인 모델 테스트")
    print("=" * 80)
    
    # 데이터 로더
    loader = DataLoader()
    test_info = loader.load_test_info()
    
    # 첫 번째 에피소드로 테스트
    game_episode = test_info.iloc[0]['game_episode']
    episode_df = loader.load_episode(game_episode)
    
    print(f"\n테스트 에피소드: {game_episode}")
    print(f"액션 수: {len(episode_df)}")
    
    # 모델 1: 단순 통계
    model1 = StatisticalBaseline(recent_weight=0.5)
    pred_x1, pred_y1 = model1.predict(episode_df)
    print(f"\n{model1}")
    print(f"  예측: ({pred_x1:.2f}, {pred_y1:.2f})")
    
    # 모델 2: 전진 바이어스
    model2 = ImprovedBaseline(recent_weight=0.7, forward_bias=0.2)
    pred_x2, pred_y2 = model2.predict(episode_df)
    print(f"\n{model2}")
    print(f"  예측: ({pred_x2:.2f}, {pred_y2:.2f})")
    
    # 마지막 패스 정보
    passes = episode_df[episode_df['type_name'] == 'Pass']
    if len(passes) > 0:
        last_pass = passes.iloc[-1]
        print(f"\n참고: 마지막 패스 end 위치")
        print(f"  ({last_pass['end_x']:.2f}, {last_pass['end_y']:.2f})")
    
    print("\n" + "=" * 80)
    print("✓ 테스트 완료!")
    print("=" * 80)