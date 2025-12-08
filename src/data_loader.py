"""
데이터 로더 모듈
테스트 데이터와 에피소드 파일을 로드하고 관리
"""

from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from src.config import DATA_DIR, TEST_CSV, SAMPLE_SUBMISSION_CSV
from src.utils import setup_logger, load_episode_safely

logger = setup_logger(__name__)


class DataLoader:
    """데이터 로딩을 담당하는 클래스"""
    
    def __init__(self, data_dir: Path = DATA_DIR):
        """
        Args:
            data_dir: 데이터 디렉토리 경로
        """
        self.data_dir = data_dir
        self.test_info: Optional[pd.DataFrame] = None
        self.sample_submission: Optional[pd.DataFrame] = None
    
    def load_test_info(self) -> pd.DataFrame:
        """
        test.csv 파일을 로드합니다.
        
        Returns:
            test.csv DataFrame
        """
        if self.test_info is None:
            logger.info(f"test.csv 로딩 중: {TEST_CSV}")
            self.test_info = pd.read_csv(TEST_CSV)
            logger.info(f"✓ {len(self.test_info)}개 에피소드 정보 로드 완료")
        
        return self.test_info
    
    def load_sample_submission(self) -> pd.DataFrame:
        """
        sample_submission.csv 파일을 로드합니다.
        
        Returns:
            sample_submission.csv DataFrame
        """
        if self.sample_submission is None:
            logger.info(f"sample_submission.csv 로딩 중: {SAMPLE_SUBMISSION_CSV}")
            self.sample_submission = pd.read_csv(SAMPLE_SUBMISSION_CSV)
            logger.info(f"✓ {len(self.sample_submission)}개 제출 템플릿 로드 완료")
        
        return self.sample_submission
    
    def load_episode(self, game_episode: str) -> Optional[pd.DataFrame]:
        """
        특정 에피소드 파일을 로드합니다.
        
        Args:
            game_episode: 에피소드 ID (예: "153363_1")
        
        Returns:
            에피소드 DataFrame 또는 None
        """
        # 방법 1: data/ 바로 아래에서 찾기
        file_path = self.data_dir / f"{game_episode}.csv"
        if file_path.exists():
            return load_episode_safely(str(file_path))
        
        # 방법 2: data/test/game_id/ 폴더에서 찾기
        game_id = game_episode.split('_')[0]  # "153363_1" -> "153363"
        file_path = self.data_dir / "test" / game_id / f"{game_episode}.csv"
        if file_path.exists():
            return load_episode_safely(str(file_path))
        
        # 방법 3: test.csv의 path 컬럼 활용
        if self.test_info is not None:
            row = self.test_info[self.test_info['game_episode'] == game_episode]
            if len(row) > 0:
                path = row.iloc[0]['path']
                # path가 "./test/153363/153363_1.csv" 형식
                file_path = self.data_dir / path.replace('./', '')
                if file_path.exists():
                    return load_episode_safely(str(file_path))
        
        logger.warning(f"에피소드 파일을 찾을 수 없음: {game_episode}")
        return None
    
    def load_all_episodes(self, show_progress: bool = True) -> dict[str, pd.DataFrame]:
        """
        모든 테스트 에피소드를 로드합니다.
        
        Args:
            show_progress: 진행 상황 표시 여부
        
        Returns:
            {game_episode: DataFrame} 딕셔너리
        """
        test_info = self.load_test_info()
        episodes = {}
        
        iterator = test_info['game_episode'].values
        if show_progress:
            iterator = tqdm(iterator, desc="에피소드 로딩")
        
        loaded_count = 0
        for game_episode in iterator:
            episode_df = self.load_episode(game_episode)
            if episode_df is not None and len(episode_df) > 0:
                episodes[game_episode] = episode_df
                loaded_count += 1
        
        logger.info(f"✓ 총 {loaded_count}/{len(test_info)}개 에피소드 로드 완료")
        return episodes
    
    def load_all_test_data(self) -> pd.DataFrame:
        """
        모든 테스트 데이터를 하나의 DataFrame으로 로드합니다.
        (LightGBM inference용)
        
        Returns:
            pd.DataFrame: 모든 테스트 에피소드 데이터
        """
        test_info = self.load_test_info()
        
        all_data = []
        
        for _, row in tqdm(test_info.iterrows(), total=len(test_info), desc="테스트 데이터 로딩"):
            game_episode = row['game_episode']
            
            # 에피소드 로드
            episode_df = self.load_episode(game_episode)
            
            if episode_df is not None and len(episode_df) > 0:
                all_data.append(episode_df)
        
        # 모든 데이터 합치기
        all_test_df = pd.concat(all_data, ignore_index=True)
        
        logger.info(f"✓ 전체 테스트 데이터 로드 완료: {len(all_test_df):,} rows")
        
        return all_test_df
    
    def get_episode_files(self) -> list[Path]:
        """
        데이터 디렉토리의 모든 에피소드 파일 경로를 반환합니다.
        
        Returns:
            에피소드 파일 경로 리스트
        """
        episode_files = []
        
        # data/ 바로 아래에서 찾기
        files = list(self.data_dir.glob("*_*.csv"))
        excluded = ["test", "sample_submission", "match_info", "train"]
        files = [f for f in files if f.stem not in excluded]
        episode_files.extend(files)
        
        # data/test/ 폴더에서 찾기
        test_dir = self.data_dir / "test"
        if test_dir.exists():
            for game_dir in test_dir.iterdir():
                if game_dir.is_dir():
                    files = list(game_dir.glob("*_*.csv"))
                    episode_files.extend(files)
        
        return episode_files


def get_episode_statistics(episode_df: pd.DataFrame) -> dict:
    """
    에피소드의 기본 통계를 계산합니다.
    
    Args:
        episode_df: 에피소드 DataFrame
    
    Returns:
        통계 딕셔너리
    """
    if episode_df is None or len(episode_df) == 0:
        return {}
    
    stats = {
        "total_actions": len(episode_df),
        "action_types": episode_df["type_name"].value_counts().to_dict(),
        "num_passes": len(episode_df[episode_df["type_name"] == "Pass"]),
        "duration": episode_df["time_seconds"].max() - episode_df["time_seconds"].min(),
        "unique_players": episode_df["player_id"].nunique(),
    }
    
    return stats


# ============================================================================
# 메인 실행 (테스트용)
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("데이터 로더 테스트")
    print("=" * 80)
    
    # DataLoader 인스턴스 생성
    loader = DataLoader()
    
    # test.csv 로드
    print("\n[1] test.csv 로드")
    test_info = loader.load_test_info()
    print(f"   에피소드 수: {len(test_info)}")
    print(f"   컬럼: {list(test_info.columns)}")
    print(f"\n   처음 3줄:")
    print(test_info.head(3))
    
    # sample_submission.csv 로드
    print("\n[2] sample_submission.csv 로드")
    submission = loader.load_sample_submission()
    print(f"   행 수: {len(submission)}")
    print(f"   컬럼: {list(submission.columns)}")
    
    # 에피소드 파일 확인
    print("\n[3] 에피소드 파일 확인")
    episode_files = loader.get_episode_files()
    print(f"   발견된 파일 수: {len(episode_files)}")
    
    if len(episode_files) > 0:
        # 첫 번째 에피소드 로드
        print(f"\n[4] 샘플 에피소드 로드")
        first_episode = test_info.iloc[0]['game_episode']
        episode_df = loader.load_episode(first_episode)
        
        if episode_df is not None:
            print(f"   에피소드: {first_episode}")
            print(f"   액션 수: {len(episode_df)}")
            print(f"   컬럼: {list(episode_df.columns)}")
            
            # 통계
            stats = get_episode_statistics(episode_df)
            print(f"\n[5] 에피소드 통계")
            print(f"   총 액션: {stats['total_actions']}")
            print(f"   패스 수: {stats['num_passes']}")
            print(f"   지속 시간: {stats['duration']:.1f}초")
            print(f"   고유 선수 수: {stats['unique_players']}")
            
            print(f"\n[6] 액션 타입 분포")
            for action_type, count in list(stats['action_types'].items())[:5]:
                print(f"      {action_type}: {count}")
    else:
        print("\n   ⚠️  에피소드 파일이 없습니다!")
        print("   → data/ 폴더에 대회 데이터를 넣어주세요.")
    
    print("\n" + "=" * 80)
    print("✓ 테스트 완료!")
    print("=" * 80)