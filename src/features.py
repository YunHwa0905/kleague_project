"""
Feature Engineering 모듈
에피소드 데이터에서 예측에 유용한 Feature 추출
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional

from src.config import FIELD_LENGTH, FIELD_WIDTH
from src.utils import calculate_distance, calculate_angle, get_field_zone


class FeatureExtractor:
    """
    에피소드에서 Feature를 추출하는 클래스
    
    추출하는 Feature:
    1. 시퀀스 기본 통계
    2. 위치 기반 Feature
    3. 방향성 Feature
    4. 시간 기반 Feature
    5. 팀/선수 정보
    """
    
    def __init__(self):
        self.feature_names = []
    
    def extract_features(self, episode_df: pd.DataFrame) -> Dict[str, float]:
        """
        에피소드에서 모든 Feature를 추출합니다.
        
        Args:
            episode_df: 에피소드 DataFrame
        
        Returns:
            features: Feature 딕셔너리
        """
        if episode_df is None or len(episode_df) == 0:
            return self._default_features()
        
        features = {}
        
        # 1. 시퀀스 기본 통계
        features.update(self._extract_sequence_features(episode_df))
        
        # 2. 위치 기반 Feature
        features.update(self._extract_location_features(episode_df))
        
        # 3. 방향성 Feature
        features.update(self._extract_direction_features(episode_df))
        
        # 4. 시간 기반 Feature
        features.update(self._extract_time_features(episode_df))
        
        # 5. 팀/선수 정보
        features.update(self._extract_team_features(episode_df))

        all_feature_names = self.get_feature_names()
        for feature_name in all_feature_names:
            if feature_name not in features:
                features[feature_name] = 0.0  # 기본값 0
        
        return features
    
    def _extract_sequence_features(self, episode_df: pd.DataFrame) -> Dict[str, float]:
        """시퀀스 기본 통계 Feature"""
        features = {}
        
        # 전체 액션 수
        features['total_actions'] = len(episode_df)
        
        # Pass 필터링
        passes = episode_df[episode_df['type_name'] == 'Pass']
        successful_passes = passes[passes['result_name'] == 'Successful']
        
        # Pass 통계
        features['num_passes'] = len(passes)
        features['num_successful_passes'] = len(successful_passes)
        features['pass_success_rate'] = len(successful_passes) / len(passes) if len(passes) > 0 else 0.0
        
        # Carry 통계
        carries = episode_df[episode_df['type_name'] == 'Carry']
        features['num_carries'] = len(carries)
        features['carry_ratio'] = len(carries) / len(episode_df) if len(episode_df) > 0 else 0.0
        
        # 기타 액션
        features['num_recoveries'] = len(episode_df[episode_df['type_name'] == 'Recovery'])
        features['num_tackles'] = len(episode_df[episode_df['type_name'] == 'Tackle'])
        features['num_interceptions'] = len(episode_df[episode_df['type_name'] == 'Interception'])
        
        # 고유 선수 수
        features['unique_players'] = episode_df['player_id'].nunique()
        
        return features
    
    def _extract_location_features(self, episode_df: pd.DataFrame) -> Dict[str, float]:
        """위치 기반 Feature"""
        features = {}
        
        # 마지막 액션 위치
        last_action = episode_df.iloc[-1]
        
        if pd.notna(last_action['end_x']) and pd.notna(last_action['end_y']):
            features['last_end_x'] = last_action['end_x']
            features['last_end_y'] = last_action['end_y']
        else:
            features['last_end_x'] = FIELD_LENGTH / 2
            features['last_end_y'] = FIELD_WIDTH / 2
        
        # 마지막 패스 위치
        passes = episode_df[episode_df['type_name'] == 'Pass']
        if len(passes) > 0:
            last_pass = passes.iloc[-1]
            if pd.notna(last_pass['end_x']) and pd.notna(last_pass['end_y']):
                features['last_pass_end_x'] = last_pass['end_x']
                features['last_pass_end_y'] = last_pass['end_y']
            else:
                features['last_pass_end_x'] = features['last_end_x']
                features['last_pass_end_y'] = features['last_end_y']
        else:
            features['last_pass_end_x'] = features['last_end_x']
            features['last_pass_end_y'] = features['last_end_y']
        
        # 필드 영역
        last_x = features['last_pass_end_x']
        last_y = features['last_pass_end_y']
        
        # X축 영역 (0: 수비, 1: 중원, 2: 공격)
        if last_x < FIELD_LENGTH / 3:
            features['field_zone_x'] = 0  # 수비
        elif last_x < 2 * FIELD_LENGTH / 3:
            features['field_zone_x'] = 1  # 중원
        else:
            features['field_zone_x'] = 2  # 공격
        
        # Y축 영역 (0: 좌측, 1: 중앙, 2: 우측)
        if last_y < FIELD_WIDTH / 3:
            features['field_zone_y'] = 0  # 좌측
        elif last_y < 2 * FIELD_WIDTH / 3:
            features['field_zone_y'] = 1  # 중앙
        else:
            features['field_zone_y'] = 2  # 우측
        
        # 골문까지 거리 (상대 골문: x=105)
        features['distance_to_goal'] = calculate_distance(
            last_x, last_y, FIELD_LENGTH, FIELD_WIDTH / 2
        )
        
        # 중앙에서 거리
        features['distance_from_center'] = calculate_distance(
            last_x, last_y, FIELD_LENGTH / 2, FIELD_WIDTH / 2
        )
        
        return features
    
    def _extract_direction_features(self, episode_df: pd.DataFrame) -> Dict[str, float]:
        """방향성 Feature"""
        features = {}
        
        passes = episode_df[episode_df['type_name'] == 'Pass'].copy()
        valid_passes = passes.dropna(subset=['start_x', 'start_y', 'end_x', 'end_y'])
        
        if len(valid_passes) == 0:
            features['avg_pass_distance'] = 0.0
            features['avg_pass_dx'] = 0.0
            features['avg_pass_dy'] = 0.0
            features['avg_pass_forward'] = 0.0
            features['total_x_progress'] = 0.0
            features['total_y_progress'] = 0.0
            return features
        
        # 패스 거리
        valid_passes['distance'] = valid_passes.apply(
            lambda row: calculate_distance(
                row['start_x'], row['start_y'],
                row['end_x'], row['end_y']
            ), axis=1
        )
        features['avg_pass_distance'] = valid_passes['distance'].mean()
        features['max_pass_distance'] = valid_passes['distance'].max()
        features['min_pass_distance'] = valid_passes['distance'].min()
        
        # 평균 이동 벡터
        valid_passes['dx'] = valid_passes['end_x'] - valid_passes['start_x']
        valid_passes['dy'] = valid_passes['end_y'] - valid_passes['start_y']
        
        features['avg_pass_dx'] = valid_passes['dx'].mean()
        features['avg_pass_dy'] = valid_passes['dy'].mean()
        
        # 전진 정도 (X축 양수 = 공격 방향)
        features['avg_pass_forward'] = valid_passes['dx'].mean()
        features['forward_pass_ratio'] = (valid_passes['dx'] > 0).sum() / len(valid_passes)
        
        # 누적 진행 (첫 액션 → 마지막 액션)
        if len(valid_passes) > 0:
            first_x = valid_passes.iloc[0]['start_x']
            first_y = valid_passes.iloc[0]['start_y']
            last_x = valid_passes.iloc[-1]['end_x']
            last_y = valid_passes.iloc[-1]['end_y']
            
            features['total_x_progress'] = last_x - first_x
            features['total_y_progress'] = last_y - first_y
        else:
            features['total_x_progress'] = 0.0
            features['total_y_progress'] = 0.0
        
        return features
    
    def _extract_time_features(self, episode_df: pd.DataFrame) -> Dict[str, float]:
        """시간 기반 Feature"""
        features = {}
        
        # 총 지속 시간
        if 'time_seconds' in episode_df.columns:
            time_min = episode_df['time_seconds'].min()
            time_max = episode_df['time_seconds'].max()
            features['duration'] = time_max - time_min
            features['avg_action_interval'] = features['duration'] / len(episode_df) if len(episode_df) > 1 else 0.0
        else:
            features['duration'] = 0.0
            features['avg_action_interval'] = 0.0
        
        return features
    
    def _extract_team_features(self, episode_df: pd.DataFrame) -> Dict[str, float]:
        """팀/선수 정보 Feature"""
        features = {}
        
        # 홈팀 여부 (마지막 액션)
        if 'is_home' in episode_df.columns:
            features['is_home'] = float(episode_df.iloc[-1]['is_home'])
        else:
            features['is_home'] = 0.5  # 알 수 없음
        
        # 팀 ID (마지막 액션)
        if 'team_id' in episode_df.columns:
            features['team_id'] = episode_df.iloc[-1]['team_id']
        else:
            features['team_id'] = 0
        
        return features
    
    def _default_features(self) -> Dict[str, float]:
        """빈 에피소드에 대한 기본 Feature"""
        return {
            'total_actions': 0,
            'num_passes': 0,
            'num_successful_passes': 0,
            'pass_success_rate': 0.0,
            'num_carries': 0,
            'carry_ratio': 0.0,
            'num_recoveries': 0,
            'num_tackles': 0,
            'num_interceptions': 0,
            'unique_players': 0,
            'last_end_x': FIELD_LENGTH / 2,
            'last_end_y': FIELD_WIDTH / 2,
            'last_pass_end_x': FIELD_LENGTH / 2,
            'last_pass_end_y': FIELD_WIDTH / 2,
            'field_zone_x': 1,
            'field_zone_y': 1,
            'distance_to_goal': FIELD_LENGTH / 2,
            'distance_from_center': 0.0,
            'avg_pass_distance': 0.0,
            'max_pass_distance': 0.0,
            'min_pass_distance': 0.0,
            'avg_pass_dx': 0.0,
            'avg_pass_dy': 0.0,
            'avg_pass_forward': 0.0,
            'forward_pass_ratio': 0.0,
            'total_x_progress': 0.0,
            'total_y_progress': 0.0,
            'duration': 0.0,
            'avg_action_interval': 0.0,
            'is_home': 0.5,
            'team_id': 0,
        }
    
    def get_feature_names(self) -> list:
        """Feature 이름 리스트 반환"""
        sample_features = self._default_features()
        return list(sample_features.keys())


# ============================================================================
# 메인 실행 (테스트용)
# ============================================================================

if __name__ == "__main__":
    from src.data_loader import DataLoader
    
    print("=" * 80)
    print("Feature Extractor 테스트")
    print("=" * 80)
    
    # 데이터 로더
    loader = DataLoader()
    test_info = loader.load_test_info()
    
    # Feature Extractor
    extractor = FeatureExtractor()
    
    # 첫 번째 에피소드 테스트
    game_episode = test_info.iloc[0]['game_episode']
    episode_df = loader.load_episode(game_episode)
    
    print(f"\n테스트 에피소드: {game_episode}")
    print(f"액션 수: {len(episode_df)}")
    
    # Feature 추출
    features = extractor.extract_features(episode_df)
    
    print(f"\n추출된 Feature 수: {len(features)}")
    print(f"\nFeature 목록:")
    for name, value in features.items():
        print(f"  {name:30s}: {value:10.2f}")
    
    print("\n" + "=" * 80)
    print("✓ 테스트 완료!")
    print("=" * 80)