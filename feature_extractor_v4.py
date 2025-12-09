"""
고급 Feature 추출기 v4
- 기존 29개 + 신규 40개 = 총 69개 Feature
- 2-3점 개선 목표
"""

import pandas as pd
import numpy as np


class FeatureExtractorV4:
    """고급 Feature를 포함한 Feature 추출기"""
    
    def extract_features(self, df_episode):
        """
        하나의 에피소드에서 Feature 추출
        
        기존 29개 + 신규 40개 = 총 69개 Feature
        """
        # 마지막 패스 제외 (end_x가 NaN인 행)
        df_clean = df_episode[df_episode['end_x'].notna()].copy()
        
        # 마지막 패스 정보 (예측을 위한 입력)
        last_pass_mask = df_episode['end_x'].isna()
        if last_pass_mask.sum() == 0:
            last_pass = df_episode.iloc[-1]
        else:
            last_pass = df_episode[last_pass_mask].iloc[0]
        
        features = {}
        
        # ============================================
        # 기존 Feature (29개)
        # ============================================
        features.update(self._extract_basic_features(df_clean, last_pass))
        
        # ============================================
        # 신규 고급 Feature (40개)
        # ============================================
        features.update(self._extract_sequence_pattern_features(df_clean, last_pass))
        features.update(self._extract_spatial_features(df_clean, last_pass))
        features.update(self._extract_temporal_features(df_clean, last_pass))
        features.update(self._extract_velocity_features(df_clean, last_pass))
        features.update(self._extract_game_situation_features(df_clean, last_pass))
        
        return features
    
    def _extract_basic_features(self, df_clean, last_pass):
        """기존 29개 Feature"""
        features = {}
        
        # 1. 마지막 패스 시작 좌표
        features['last_start_x'] = float(last_pass['start_x'])
        features['last_start_y'] = float(last_pass['start_y'])
        features['last_time'] = float(last_pass['time_seconds'])
        
        # 2. 기본 정보
        features['is_home'] = int(last_pass['is_home'])
        features['period_id'] = int(last_pass['period_id'])
        features['episode_length'] = len(df_clean)
        
        # 3. 시퀀스 통계
        if len(df_clean) > 0:
            features['seq_mean_x'] = float(df_clean['end_x'].mean())
            features['seq_mean_y'] = float(df_clean['end_y'].mean())
            features['seq_std_x'] = float(df_clean['end_x'].std()) if len(df_clean) > 1 else 0.0
            features['seq_std_y'] = float(df_clean['end_y'].std()) if len(df_clean) > 1 else 0.0
            
            first_x = float(df_clean.iloc[0]['end_x'])
            first_y = float(df_clean.iloc[0]['end_y'])
            prev_x = float(df_clean.iloc[-1]['end_x'])
            prev_y = float(df_clean.iloc[-1]['end_y'])
            
            features['progress_x'] = prev_x - first_x
            features['progress_y'] = prev_y - first_y
            
            try:
                features['progress_dist'] = float(np.sqrt(features['progress_x']**2 + features['progress_y']**2))
            except:
                features['progress_dist'] = 0.0
            
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
        
        # 4. 최근 패턴
        recent_passes = df_clean[df_clean['type_name'] == 'Pass'].tail(3)
        
        if len(recent_passes) > 0:
            moves_x = (recent_passes['end_x'] - recent_passes['start_x']).values
            moves_y = (recent_passes['end_y'] - recent_passes['start_y']).values
            moves_x = np.array([float(x) for x in moves_x])
            moves_y = np.array([float(y) for y in moves_y])
            
            features['recent_pass_count'] = len(recent_passes)
            features['recent_avg_move_x'] = float(np.mean(moves_x))
            features['recent_avg_move_y'] = float(np.mean(moves_y))
            
            try:
                distances = np.sqrt(moves_x**2 + moves_y**2)
                features['recent_avg_distance'] = float(np.mean(distances))
            except:
                features['recent_avg_distance'] = 0.0
            
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
        
        # 5. 액션 타입
        action_counts = df_clean['type_name'].value_counts()
        features['pass_count'] = int(action_counts.get('Pass', 0))
        features['carry_count'] = int(action_counts.get('Carry', 0))
        features['pass_ratio'] = float(features['pass_count'] / len(df_clean)) if len(df_clean) > 0 else 0.0
        
        successful = df_clean[df_clean['result_name'] == 'Successful']
        features['success_rate'] = float(len(successful) / len(df_clean)) if len(df_clean) > 0 else 0.0
        
        # 6. 필드 위치
        start_x = float(last_pass['start_x'])
        start_y = float(last_pass['start_y'])
        
        features['start_zone_x'] = int(start_x / 35)
        features['start_zone_y'] = int(start_y / 22.67)
        features['dist_from_center_x'] = abs(start_x - 52.5)
        features['dist_from_center_y'] = abs(start_y - 34)
        
        return features
    
    def _extract_sequence_pattern_features(self, df_clean, last_pass):
        """시퀀스 패턴 Feature (10개)"""
        features = {}
        
        passes = df_clean[df_clean['type_name'] == 'Pass'].copy()
        
        if len(passes) >= 3:
            # 최근 N개 패스의 방향 변화
            for n in [3, 5, 10]:
                recent_n = passes.tail(min(n, len(passes)))
                if len(recent_n) >= 2:
                    angles = []
                    for i in range(len(recent_n) - 1):
                        dx1 = recent_n.iloc[i]['end_x'] - recent_n.iloc[i]['start_x']
                        dy1 = recent_n.iloc[i]['end_y'] - recent_n.iloc[i]['start_y']
                        dx2 = recent_n.iloc[i+1]['end_x'] - recent_n.iloc[i+1]['start_x']
                        dy2 = recent_n.iloc[i+1]['end_y'] - recent_n.iloc[i+1]['start_y']
                        
                        try:
                            angle1 = np.arctan2(dy1, dx1)
                            angle2 = np.arctan2(dy2, dx2)
                            angle_change = abs(angle2 - angle1)
                            if angle_change > np.pi:
                                angle_change = 2 * np.pi - angle_change
                            angles.append(angle_change)
                        except:
                            pass
                    
                    features[f'angle_change_last_{n}'] = float(np.mean(angles)) if len(angles) > 0 else 0.0
                else:
                    features[f'angle_change_last_{n}'] = 0.0
            
            # 패스 방향의 일관성
            if len(passes) >= 3:
                dx_list = (passes['end_x'] - passes['start_x']).values[-5:]
                dy_list = (passes['end_y'] - passes['start_y']).values[-5:]
                features['pass_direction_consistency'] = float(1.0 / (1.0 + np.std(dx_list) + np.std(dy_list)))
            else:
                features['pass_direction_consistency'] = 0.0
        else:
            features['angle_change_last_3'] = 0.0
            features['angle_change_last_5'] = 0.0
            features['angle_change_last_10'] = 0.0
            features['pass_direction_consistency'] = 0.0
        
        # 연속 성공 패스
        if len(passes) > 0:
            consecutive_success = 0
            for i in range(len(passes) - 1, -1, -1):
                if passes.iloc[i]['result_name'] == 'Successful':
                    consecutive_success += 1
                else:
                    break
            features['consecutive_successful_passes'] = consecutive_success
        else:
            features['consecutive_successful_passes'] = 0
        
        # 짧은/긴 패스 비율
        if len(passes) > 0:
            distances = []
            for _, row in passes.iterrows():
                try:
                    dist = np.sqrt((row['end_x'] - row['start_x'])**2 + (row['end_y'] - row['start_y'])**2)
                    distances.append(dist)
                except:
                    pass
            
            if len(distances) > 0:
                short_passes = sum(1 for d in distances if d < 10)
                long_passes = sum(1 for d in distances if d > 20)
                features['short_pass_ratio'] = short_passes / len(distances)
                features['long_pass_ratio'] = long_passes / len(distances)
            else:
                features['short_pass_ratio'] = 0.0
                features['long_pass_ratio'] = 0.0
        else:
            features['short_pass_ratio'] = 0.0
            features['long_pass_ratio'] = 0.0
        
        # 측면/중앙 패스 비율
        if len(passes) > 0:
            side_passes = sum(1 for _, row in passes.iterrows() 
                            if row['start_y'] < 20 or row['start_y'] > 48)
            features['side_pass_ratio'] = side_passes / len(passes)
        else:
            features['side_pass_ratio'] = 0.0
        
        # 전진/후진 패스 비율
        if len(passes) > 0:
            forward_passes = sum(1 for _, row in passes.iterrows() 
                               if (row['end_x'] - row['start_x']) > 0)
            features['forward_pass_ratio'] = forward_passes / len(passes)
        else:
            features['forward_pass_ratio'] = 0.0
        
        return features
    
    def _extract_spatial_features(self, df_clean, last_pass):
        """공간 정보 Feature (8개)"""
        features = {}
        
        start_x = float(last_pass['start_x'])
        start_y = float(last_pass['start_y'])
        
        # 상대 골문까지 거리/각도
        goal_x, goal_y = 105, 34
        features['dist_to_goal'] = float(np.sqrt((goal_x - start_x)**2 + (goal_y - start_y)**2))
        features['angle_to_goal'] = float(np.arctan2(goal_y - start_y, goal_x - start_x))
        
        # 터치라인까지 거리
        features['dist_to_top_line'] = float(abs(68 - start_y))
        features['dist_to_bottom_line'] = float(abs(0 - start_y))
        features['dist_to_nearest_sideline'] = float(min(start_y, 68 - start_y))
        
        # 골라인까지 거리
        features['dist_to_goal_line'] = float(abs(105 - start_x))
        
        # 페널티 박스 안/밖
        in_penalty_box = 1 if (start_x > 88.5 and 13.84 < start_y < 54.16) else 0
        features['in_penalty_box'] = in_penalty_box
        
        # 필드를 9구역으로 나눔
        zone_x = int(min(start_x // 35, 2))  # 0, 1, 2
        zone_y = int(min(start_y // 22.67, 2))  # 0, 1, 2
        features['field_zone_9'] = zone_x * 3 + zone_y  # 0~8
        
        return features
    
    def _extract_temporal_features(self, df_clean, last_pass):
        """시간 패턴 Feature (6개)"""
        features = {}
        
        time_seconds = float(last_pass['time_seconds'])
        
        # 경기 시간대
        features['game_minute'] = int(time_seconds // 60)
        features['is_first_half'] = 1 if time_seconds < 2700 else 0  # 45분
        features['is_last_10_min'] = 1 if (time_seconds % 2700) > 2100 else 0  # 마지막 10분
        
        # 시간 압박 (빠른 패스 연속)
        if len(df_clean) >= 3:
            time_diffs = df_clean['time_seconds'].diff().dropna().values[-5:]
            features['avg_time_between_actions'] = float(np.mean(time_diffs)) if len(time_diffs) > 0 else 0.0
            features['is_fast_tempo'] = 1 if features['avg_time_between_actions'] < 3 else 0
        else:
            features['avg_time_between_actions'] = 0.0
            features['is_fast_tempo'] = 0
        
        # 에피소드 지속 시간
        if len(df_clean) > 0:
            duration = time_seconds - df_clean.iloc[0]['time_seconds']
            features['episode_duration'] = float(duration)
        else:
            features['episode_duration'] = 0.0
        
        return features
    
    def _extract_velocity_features(self, df_clean, last_pass):
        """속도/가속도 Feature (8개)"""
        features = {}
        
        passes = df_clean[df_clean['type_name'] == 'Pass'].copy()
        
        if len(passes) >= 2:
            # 패스 속도 (거리/시간)
            velocities = []
            accelerations = []
            
            for i in range(len(passes) - 1):
                try:
                    dist = np.sqrt((passes.iloc[i+1]['start_x'] - passes.iloc[i]['end_x'])**2 +
                                 (passes.iloc[i+1]['start_y'] - passes.iloc[i]['end_y'])**2)
                    time_diff = passes.iloc[i+1]['time_seconds'] - passes.iloc[i]['time_seconds']
                    
                    if time_diff > 0:
                        velocity = dist / time_diff
                        velocities.append(velocity)
                        
                        if len(velocities) >= 2:
                            accel = (velocities[-1] - velocities[-2]) / time_diff
                            accelerations.append(accel)
                except:
                    pass
            
            features['avg_pass_velocity'] = float(np.mean(velocities)) if len(velocities) > 0 else 0.0
            features['max_pass_velocity'] = float(np.max(velocities)) if len(velocities) > 0 else 0.0
            features['velocity_std'] = float(np.std(velocities)) if len(velocities) > 1 else 0.0
            features['avg_acceleration'] = float(np.mean(accelerations)) if len(accelerations) > 0 else 0.0
            features['is_accelerating'] = 1 if features['avg_acceleration'] > 0 else 0
        else:
            features['avg_pass_velocity'] = 0.0
            features['max_pass_velocity'] = 0.0
            features['velocity_std'] = 0.0
            features['avg_acceleration'] = 0.0
            features['is_accelerating'] = 0
        
        # X/Y 방향 속도
        if len(passes) >= 2:
            dx = passes['end_x'].values[-1] - passes['start_x'].values[0]
            dy = passes['end_y'].values[-1] - passes['start_y'].values[0]
            dt = passes['time_seconds'].values[-1] - passes['time_seconds'].values[0]
            
            if dt > 0:
                features['velocity_x'] = float(dx / dt)
                features['velocity_y'] = float(dy / dt)
            else:
                features['velocity_x'] = 0.0
                features['velocity_y'] = 0.0
        else:
            features['velocity_x'] = 0.0
            features['velocity_y'] = 0.0
        
        return features
    
    def _extract_game_situation_features(self, df_clean, last_pass):
        """경기 상황 Feature (8개)"""
        features = {}
        
        # 공격/수비 모드 판단
        if len(df_clean) > 0:
            avg_x = df_clean['end_x'].mean()
            features['is_attacking_mode'] = 1 if avg_x > 52.5 else 0
            features['attack_intensity'] = float((avg_x - 52.5) / 52.5)  # -1 ~ 1
        else:
            features['is_attacking_mode'] = 0
            features['attack_intensity'] = 0.0
        
        # 압박 상황 (빠른 액션 연속)
        if len(df_clean) >= 3:
            features['action_density'] = float(len(df_clean) / (df_clean['time_seconds'].max() - df_clean['time_seconds'].min() + 0.01))
        else:
            features['action_density'] = 0.0
        
        # 같은 팀 연속 액션
        if 'team_id' in df_clean.columns and len(df_clean) > 0:
            same_team_count = 1
            last_team = df_clean.iloc[-1]['team_id']
            for i in range(len(df_clean) - 2, -1, -1):
                if df_clean.iloc[i]['team_id'] == last_team:
                    same_team_count += 1
                else:
                    break
            features['same_team_sequence'] = same_team_count
        else:
            features['same_team_sequence'] = 1
        
        # 필드 커버리지 (활동 범위)
        if len(df_clean) > 0:
            x_range = df_clean['end_x'].max() - df_clean['end_x'].min()
            y_range = df_clean['end_y'].max() - df_clean['end_y'].min()
            features['field_coverage_x'] = float(x_range)
            features['field_coverage_y'] = float(y_range)
            features['field_coverage_area'] = float(x_range * y_range)
        else:
            features['field_coverage_x'] = 0.0
            features['field_coverage_y'] = 0.0
            features['field_coverage_area'] = 0.0
        
        # 마지막 액션과의 시간 차이
        if len(df_clean) > 0:
            time_since_last = float(last_pass['time_seconds'] - df_clean.iloc[-1]['time_seconds'])
            features['time_since_last_action'] = time_since_last
        else:
            features['time_since_last_action'] = 0.0
        
        return features


if __name__ == "__main__":
    print("Feature Extractor V4 - Advanced Features")
    print("=" * 60)
    print("기존 29개 + 신규 40개 = 총 69개 Feature!")