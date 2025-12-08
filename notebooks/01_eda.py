"""
Step 1: 데이터 탐색 (EDA)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.data_loader import DataLoader, get_episode_statistics
from src.config import OUTPUT_DIR, FIELD_LENGTH, FIELD_WIDTH

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 스타일
sns.set_style("whitegrid")

def analyze_episodes(loader: DataLoader, num_samples: int = 100):
    """에피소드들을 분석합니다."""
    
    print("=" * 80)
    print("K리그 패스 예측 - 데이터 탐색")
    print("=" * 80)
    
    # 테스트 정보 로드
    test_info = loader.load_test_info()
    print(f"\n📊 전체 데이터")
    print(f"   테스트 에피소드 수: {len(test_info):,}")
    
    # 샘플 에피소드 로드
    print(f"\n📥 샘플 {num_samples}개 에피소드 분석 중...")
    sample_episodes = test_info.head(num_samples)['game_episode'].values
    
    all_data = []
    episode_stats = []
    
    for game_episode in sample_episodes:
        episode_df = loader.load_episode(game_episode)
        if episode_df is not None and len(episode_df) > 0:
            all_data.append(episode_df)
            stats = get_episode_statistics(episode_df)
            stats['game_episode'] = game_episode
            episode_stats.append(stats)
    
    # 하나로 합치기
    combined = pd.concat(all_data, ignore_index=True)
    stats_df = pd.DataFrame(episode_stats)
    
    print(f"\n✓ {len(all_data)}개 에피소드 분석 완료")
    print(f"   총 액션 수: {len(combined):,}")
    
    # 기본 통계
    print(f"\n📈 에피소드 통계")
    print(f"   평균 액션 수: {stats_df['total_actions'].mean():.1f}")
    print(f"   평균 패스 수: {stats_df['num_passes'].mean():.1f}")
    print(f"   평균 지속 시간: {stats_df['duration'].mean():.1f}초")
    print(f"   평균 고유 선수 수: {stats_df['unique_players'].mean():.1f}")
    
    # 액션 타입 분포
    print(f"\n🎯 액션 타입 분포")
    action_counts = combined['type_name'].value_counts()
    for action_type, count in action_counts.head(10).items():
        pct = count / len(combined) * 100
        print(f"   {action_type:20s}: {count:5,} ({pct:5.1f}%)")
    
    # 좌표 범위
    print(f"\n📍 좌표 통계")
    print(f"   start_x: [{combined['start_x'].min():.1f}, {combined['start_x'].max():.1f}]")
    print(f"   start_y: [{combined['start_y'].min():.1f}, {combined['start_y'].max():.1f}]")
    print(f"   end_x: [{combined['end_x'].min():.1f}, {combined['end_x'].max():.1f}]")
    print(f"   end_y: [{combined['end_y'].min():.1f}, {combined['end_y'].max():.1f}]")
    
    # 패스 분석
    passes = combined[combined['type_name'] == 'Pass'].copy()
    passes['distance'] = np.sqrt(
        (passes['end_x'] - passes['start_x'])**2 + 
        (passes['end_y'] - passes['start_y'])**2
    )
    
    print(f"\n⚽ 패스 통계")
    print(f"   총 패스 수: {len(passes):,}")
    print(f"   성공한 패스: {len(passes[passes['result_name'] == 'Successful']):,}")
    print(f"   평균 패스 거리: {passes['distance'].mean():.1f}m")
    print(f"   중앙값 패스 거리: {passes['distance'].median():.1f}m")
    
    return combined, stats_df, passes


def create_visualizations(combined, stats_df, passes, output_dir: Path):
    """시각화를 생성합니다."""
    
    print(f"\n🎨 시각화 생성 중...")
    
    fig = plt.figure(figsize=(18, 12))
    
    # 1. 에피소드 길이 분포
    ax1 = plt.subplot(3, 3, 1)
    ax1.hist(stats_df['total_actions'], bins=20, edgecolor='black', alpha=0.7)
    ax1.axvline(stats_df['total_actions'].mean(), color='red', 
                linestyle='--', label=f'평균: {stats_df["total_actions"].mean():.1f}')
    ax1.set_xlabel('에피소드 길이 (액션 수)')
    ax1.set_ylabel('빈도')
    ax1.set_title('에피소드 길이 분포')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 액션 타입 분포
    ax2 = plt.subplot(3, 3, 2)
    action_counts = combined['type_name'].value_counts().head(10)
    ax2.barh(range(len(action_counts)), action_counts.values)
    ax2.set_yticks(range(len(action_counts)))
    ax2.set_yticklabels(action_counts.index)
    ax2.set_xlabel('개수')
    ax2.set_title('액션 타입 분포 (Top 10)')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. 패스 거리 분포
    ax3 = plt.subplot(3, 3, 3)
    ax3.hist(passes['distance'], bins=30, edgecolor='black', alpha=0.7)
    ax3.axvline(passes['distance'].mean(), color='red', 
                linestyle='--', label=f'평균: {passes["distance"].mean():.1f}')
    ax3.set_xlabel('패스 거리 (m)')
    ax3.set_ylabel('빈도')
    ax3.set_title('패스 거리 분포')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 시작 위치 히트맵
    ax4 = plt.subplot(3, 3, 4)
    h = ax4.hexbin(combined['start_x'], combined['start_y'], 
                   gridsize=30, cmap='YlOrRd', alpha=0.8)
    ax4.set_xlabel('X (0-105)')
    ax4.set_ylabel('Y (0-68)')
    ax4.set_title('시작 위치 분포')
    ax4.set_xlim(0, FIELD_LENGTH)
    ax4.set_ylim(0, FIELD_WIDTH)
    plt.colorbar(h, ax=ax4)
    
    # 5. 종료 위치 히트맵
    ax5 = plt.subplot(3, 3, 5)
    h = ax5.hexbin(combined['end_x'], combined['end_y'], 
                   gridsize=30, cmap='YlGnBu', alpha=0.8)
    ax5.set_xlabel('X (0-105)')
    ax5.set_ylabel('Y (0-68)')
    ax5.set_title('종료 위치 분포')
    ax5.set_xlim(0, FIELD_LENGTH)
    ax5.set_ylim(0, FIELD_WIDTH)
    plt.colorbar(h, ax=ax5)
    
    # 6. 패스 성공률
    ax6 = plt.subplot(3, 3, 6)
    result_counts = passes['result_name'].value_counts()
    ax6.pie(result_counts.values, labels=result_counts.index, autopct='%1.1f%%')
    ax6.set_title('패스 성공률')
    
    # 7. X축 진행 방향
    ax7 = plt.subplot(3, 3, 7)
    passes['x_progress'] = passes['end_x'] - passes['start_x']
    ax7.hist(passes['x_progress'], bins=40, edgecolor='black', alpha=0.7)
    ax7.axvline(0, color='red', linestyle='--')
    ax7.set_xlabel('X축 이동 거리 (m)')
    ax7.set_ylabel('빈도')
    ax7.set_title('X축 진행 방향 (+ = 공격 방향)')
    ax7.grid(True, alpha=0.3)
    
    # 8. Y축 진행 방향
    ax8 = plt.subplot(3, 3, 8)
    passes['y_progress'] = passes['end_y'] - passes['start_y']
    ax8.hist(passes['y_progress'], bins=40, edgecolor='black', alpha=0.7)
    ax8.axvline(0, color='red', linestyle='--')
    ax8.set_xlabel('Y축 이동 거리 (m)')
    ax8.set_ylabel('빈도')
    ax8.set_title('Y축 진행 방향')
    ax8.grid(True, alpha=0.3)
    
    # 9. 시간대별 액션 분포
    ax9 = plt.subplot(3, 3, 9)
    ax9.hist(combined['time_seconds'], bins=30, edgecolor='black', alpha=0.7)
    ax9.set_xlabel('시간 (초)')
    ax9.set_ylabel('빈도')
    ax9.set_title('시간대별 액션 분포')
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 저장
    output_path = output_dir / '01_eda_overview.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ 저장 완료: {output_path}")
    
    plt.close()


def main():
    # 데이터 로더
    loader = DataLoader()
    
    # 분석
    combined, stats_df, passes = analyze_episodes(loader, num_samples=100)
    
    # 시각화
    create_visualizations(combined, stats_df, passes, OUTPUT_DIR)
    
    print("\n" + "=" * 80)
    print("✓ EDA 완료!")
    print("=" * 80)
    print(f"\n다음 단계: 베이스라인 모델 만들기")


if __name__ == "__main__":
    main()