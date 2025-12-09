"""
수정된 Feature 추출기 v3.1
- 타입 안전성 개선
- np.sqrt 에러 수정
"""

import pandas as pd
import numpy as np


class FeatureExtractorV3:
    """Episode 내부 정보만 사용하는 올바른 Feature 추출"""
    
    def extract_features(self, df_episode):
        """
        하나의 에피소드에서 Feature 추출
        
        중요: 마지막 패스(end_x, end_y가 NaN)는 완전히 제외하고
              그 이전 액션들만 사용
        """
        # 마지막 패스 제외 (end_x가 NaN인 행)
        df_clean = df_episode[df_episode['end_x'].notna()].copy()
        
        # 마지막 패스 정보 (예측을 위한 입력)
        last_pass_mask = df_episode['end_x'].isna()
        if last_pass_mask.sum() == 0:
            # NaN이 없으면 마지막 행을 사용
            last_pass = df_episode.iloc[-1]
        else:
            last_pass = df_episode[last_pass_mask].iloc[0]
        
        features = {}
        
        # ============================================
        # 1. 마지막 패스의 시작 좌표 (가장 중요!)
        # ============================================
        features['last_start_x'] = float(last_pass['start_x'])
        features['last_start_y'] = float(last_pass['start_y'])
        features['last_time'] = float(last_pass['time_seconds'])
        
        # ============================================
        # 2. 기본 정보
        # ============================================
        features['is_home'] = int(last_pass['is_home'])
        features['period_id'] = int(last_pass['period_id'])
        features['episode_length'] = len(df_clean)
        
        # ============================================
        # 3. 시퀀스 통계 (마지막 패스 이전까지만)
        # ============================================
        if len(df_clean) > 0:
            # 전체 시퀀스의 평균 좌표
            features['seq_mean_x'] = float(df_clean['end_x'].mean())
            features['seq_mean_y'] = float(df_clean['end_y'].mean())
            features['seq_std_x'] = float(df_clean['end_x'].std()) if len(df_clean) > 1 else 0.0
            features['seq_std_y'] = float(df_clean['end_y'].std()) if len(df_clean) > 1 else 0.0
            
            # 시퀀스의 진행 방향 (첫 액션 -> 마지막 전 액션)
            first_x = float(df_clean.iloc[0]['end_x'])
            first_y = float(df_clean.iloc[0]['end_y'])
            prev_x = float(df_clean.iloc[-1]['end_x'])
            prev_y = float(df_clean.iloc[-1]['end_y'])
            
            features['progress_x'] = prev_x - first_x
            features['progress_y'] = prev_y - first_y
            
            # 안전한 sqrt 계산
            try:
                features['progress_dist'] = float(np.sqrt(features['progress_x']**2 + features['progress_y']**2))
            except:
                features['progress_dist'] = 0.0
            
            # 마지막 전 액션의 위치
            features['prev_end_x'] = prev_x
            features['prev_end_y'] = prev_y
        else:
            features['seq_mean_x'] = float(last_pass['start_x'])
            features['seq_mean_y'] = float(last_pass['start_y'])
            features['seq_std_x'] = 0.0
            features['seq_std_y'] = 0.0
            features['progress_x'] = 0.0
            features['progress_y'] = 0.0
            features['progress_dist'] = 0.0
            features['prev_end_x'] = float(last_pass['start_x'])
            features['prev_end_y'] = float(last_pass['start_y'])
        
        # ============================================
        # 4. 최근 N개 액션의 패턴
        # ============================================
        # 최근 3개 패스만 추출
        recent_passes = df_clean[df_clean['type_name'] == 'Pass'].tail(3)
        
        if len(recent_passes) > 0:
            # 최근 패스들의 평균 이동
            moves_x = (recent_passes['end_x'] - recent_passes['start_x']).values
            moves_y = (recent_passes['end_y'] - recent_passes['start_y']).values
            
            # 타입 안전하게 변환
            moves_x = np.array([float(x) for x in moves_x])
            moves_y = np.array([float(y) for y in moves_y])
            
            features['recent_pass_count'] = len(recent_passes)
            features['recent_avg_move_x'] = float(np.mean(moves_x))
            features['recent_avg_move_y'] = float(np.mean(moves_y))
            
            # 안전한 sqrt
            try:
                distances = np.sqrt(moves_x**2 + moves_y**2)
                features['recent_avg_distance'] = float(np.mean(distances))
            except:
                features['recent_avg_distance'] = 0.0
            
            # 가장 최근 패스
            if len(recent_passes) >= 1:
                last_recent = recent_passes.iloc[-1]
                features['last_recent_move_x'] = float(last_recent['end_x'] - last_recent['start_x'])
                features['last_recent_move_y'] = float(last_recent['end_y'] - last_recent['start_y'])
            else:
                features['last_recent_move_x'] = 0.0
                features['last_recent_move_y'] = 0.0
        else:
            features['recent_pass_count'] = 0
            features['recent_avg_move_x'] = 0.0
            features['recent_avg_move_y'] = 0.0
            features['recent_avg_distance'] = 0.0
            features['last_recent_move_x'] = 0.0
            features['last_recent_move_y'] = 0.0
        
        # ============================================
        # 5. 액션 타입 통계
        # ============================================
        action_counts = df_clean['type_name'].value_counts()
        features['pass_count'] = int(action_counts.get('Pass', 0))
        features['carry_count'] = int(action_counts.get('Carry', 0))
        features['pass_ratio'] = float(features['pass_count'] / len(df_clean)) if len(df_clean) > 0 else 0.0
        
        # 성공률
        successful = df_clean[df_clean['result_name'] == 'Successful']
        features['success_rate'] = float(len(successful) / len(df_clean)) if len(df_clean) > 0 else 0.0
        
        # ============================================
        # 6. 필드 위치 특성
        # ============================================
        # 마지막 패스 시작점이 필드의 어디인지
        start_x = float(last_pass['start_x'])
        start_y = float(last_pass['start_y'])
        
        features['start_zone_x'] = int(start_x / 35)  # 0(수비), 1(중앙), 2(공격)
        features['start_zone_y'] = int(start_y / 22.67)  # 0(좌), 1(중), 2(우)
        
        # 필드 중앙으로부터의 거리
        features['dist_from_center_x'] = abs(start_x - 52.5)
        features['dist_from_center_y'] = abs(start_y - 34)
        
        return features
    
    def extract_batch(self, episodes_dict):
        """여러 에피소드에서 Feature 추출"""
        features_list = []
        game_episodes = []
        
        for game_episode, df_episode in episodes_dict.items():
            try:
                features = self.extract_features(df_episode)
                features_list.append(features)
                game_episodes.append(game_episode)
            except Exception as e:
                print(f"Warning: Failed to extract features for {game_episode}: {e}")
                continue
        
        df_features = pd.DataFrame(features_list)
        df_features['game_episode'] = game_episodes
        
        return df_features


if __name__ == "__main__":
    # 테스트
    print("Feature Extractor V3.1 - Type Safe!")
    print("=" * 60)
    
    # 샘플 데이터로 테스트
    import sys
    
    try:
        df_sample = pd.read_csv('/mnt/project/153363_1.csv')
        
        extractor = FeatureExtractorV3()
        features = extractor.extract_features(df_sample)
        
        print("\n추출된 Feature:")
        for k, v in features.items():
            print(f"  {k:30s}: {v} (type: {type(v).__name__})")
        
        print(f"\n총 Feature 개수: {len(features)}")
    except Exception as e:
        print(f"Test failed: {e}")