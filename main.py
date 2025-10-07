#!/usr/bin/env python3
"""
야구 투구 비디오 분석 메인 스크립트

이 스크립트는 야구 투구 비디오를 분석하여 각 투구의 팔꿈치 각도를 측정하고,
분석 결과를 CSV 파일과 주석 처리된 비디오로 저장합니다.
"""

import os
import sys
import pandas as pd
from typing import List, Dict

# 프로젝트 모듈들 임포트
from pose_detector import PoseDetector
from pitcher_identifier import PitcherIdentifier
from elbow_angle_calculator import ElbowAngleCalculator
from release_detector import ReleaseDetector
from video_processor import VideoProcessor


class PitchAnalysisApp:
    """
    야구 투구 분석 애플리케이션
    """

    def __init__(self):
        """
        애플리케이션 초기화
        """
        print("야구 투구 분석 시스템을 초기화합니다...")

        # 각 컴포넌트 초기화
        self.pose_detector = PoseDetector(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.pitcher_identifier = PitcherIdentifier(self.pose_detector)
        self.elbow_calculator = ElbowAngleCalculator()
        self.release_detector = ReleaseDetector(self.elbow_calculator)
        self.video_processor = VideoProcessor(
            self.pose_detector,
            self.pitcher_identifier,
            self.elbow_calculator,
            self.release_detector
        )

        print("모든 컴포넌트 초기화 완료")

    def run_analysis(self, input_dir: str, output_dir: str = "output_videos") -> str:
        """
        전체 분석 실행

        Args:
            input_dir: 입력 비디오 디렉토리 경로
            output_dir: 출력 디렉토리 경로

        Returns:
            결과 CSV 파일 경로
        """
        print(f"\n분석을 시작합니다...")
        print(f"입력 디렉토리: {input_dir}")
        print(f"출력 디렉토리: {output_dir}")

        # 입력 디렉토리 확인
        if not os.path.exists(input_dir):
            print(f"오류: 입력 디렉토리가 존재하지 않습니다: {input_dir}")
            return None

        # 비디오 파일 찾기 및 처리
        video_files = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]
        video_files.sort()  # 파일명 순서대로 정렬

        if not video_files:
            print(f"오류: 입력 디렉토리에서 MP4 파일을 찾을 수 없습니다: {input_dir}")
            return None

        print(f"{len(video_files)}개의 비디오 파일을 발견했습니다")

        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)

        # 분석 결과 저장
        analysis_results = []

        for video_file in video_files:
            print(f"\n{'='*60}")
            print(f"처리 중: {video_file}")
            print(f"{'='*60}")

            # 파일명에서 투구 정보 추출
            pitch_info = self.video_processor.extract_pitch_info_from_filename(video_file)

            # 비디오 처리
            input_path = os.path.join(input_dir, video_file)
            output_path = os.path.join(output_dir, video_file.replace('.mp4', '_processed.mp4'))

            result = self.video_processor.process_video(input_path, output_path, show_preview=False)

            # 결과 정리
            if 'error' not in result:
                print(f"비디오 처리 완료: {video_file}")

                # 투구 분석 결과 생성
                pitch_result = {
                    'pitch_number': pitch_info['pitch_number'],
                    'pitch_type': pitch_info['pitch_type'],
                    'velocity_mph': pitch_info['velocity_mph'],
                    'elbow_angle_degrees': result.get('final_elbow_angle', 'N/A'),
                    'throwing_arm': result.get('throwing_arm', 'UNKNOWN'),
                    'total_frames': result.get('total_frames', 0),
                    'fps': result.get('fps', 0),
                    'processing_success': True
                }

                # 릴리스 분석 결과 추가
                if result.get('release_analysis'):
                    release_info = result['release_analysis']
                    pitch_result.update({
                        'release_frame': release_info.get('release_frame'),
                        'motion_quality': release_info.get('motion_quality', 'UNKNOWN')
                    })

                analysis_results.append(pitch_result)

            else:
                print(f"✗ 비디오 처리 실패: {video_file}")
                print(f"  오류: {result['error']}")

                # 실패한 경우도 기록
                pitch_result = {
                    'pitch_number': pitch_info['pitch_number'],
                    'pitch_type': pitch_info['pitch_type'],
                    'velocity_mph': pitch_info['velocity_mph'],
                    'elbow_angle_degrees': 'Error',
                    'throwing_arm': 'UNKNOWN',
                    'total_frames': 0,
                    'fps': 0,
                    'processing_success': False,
                    'error_message': result['error']
                }
                analysis_results_results.append(pitch_result)

        # 결과 DataFrame 생성 및 CSV 저장
        if analysis_results:
            df = pd.DataFrame(analysis_results)

            # CSV 파일 경로
            csv_path = os.path.join(output_dir, 'pitch_analysis_results.csv')

            # 결과를 CSV로 저장
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')

            print("\n분석 완료!")
            print(f"결과 CSV 파일 저장: {csv_path}")
            print(f"처리된 비디오 파일들 저장: {output_dir}")
            print(f"총 처리된 투구 수: {len(analysis_results)}")

            # 결과 요약 출력
            self._print_analysis_summary(df)

            return csv_path

        else:
            print("\n분석 실패: 처리된 비디오가 없습니다.")
            return None

    def _print_analysis_summary(self, df: pd.DataFrame):
        """
        분석 결과 요약 출력

        Args:
            df: 분석 결과 DataFrame
        """
        print("\n분석 결과 요약:")
        print(f"  총 투구 수: {len(df)}")

        # 성공한 투구 수
        successful = len(df[df['elbow_angle_degrees'] != 'Error'])
        print(f"  성공한 분석: {successful}")
        print(f"  실패한 분석: {len(df) - successful}")

        if successful > 0:
            # 팔꿈치 각도 통계 (숫자만 필터링)
            valid_angles = df[
                (df['elbow_angle_degrees'] != 'Error') &
                (df['elbow_angle_degrees'] != 'N/A') &
                (pd.to_numeric(df['elbow_angle_degrees'], errors='coerce').notna())
            ]['elbow_angle_degrees']

            if not valid_angles.empty:
                angles_numeric = pd.to_numeric(valid_angles)
                print(f"  팔꿈치 각도 평균: {angles_numeric.mean():.1f}°")
                print(f"  팔꿈치 각도 범위: {angles_numeric.min():.1f}° - {angles_numeric.max():.1f}°")

            # 구종별 분포
            pitch_types = df['pitch_type'].value_counts()
            print("  구종별 분포:")
            for pitch_type, count in pitch_types.items():
                print(f"    {pitch_type}: {count}개")

    def run_single_video_test(self, video_path: str, output_path: str = None) -> Dict:
        """
        단일 비디오 테스트 실행

        Args:
            video_path: 테스트할 비디오 파일 경로
            output_path: 출력 파일 경로 (선택사항)

        Returns:
            분석 결과
        """
        print(f"단일 비디오 테스트: {video_path}")

        if not output_path:
            output_path = video_path.replace('.mp4', '_test_processed.mp4')

        result = self.video_processor.process_video(video_path, output_path, show_preview=True)

        if 'error' not in result:
            print("테스트 완료")
        else:
            print(f"테스트 실패: {result['error']}")

        return result


def main():
    """
    메인 함수
    """
    print("야구 투구 분석 시스템")
    print("=" * 50)

    # 입력 인자 처리
    if len(sys.argv) < 2:
        print("사용법:")
        print("  python main.py <입력_디렉토리> [출력_디렉토리]")
        print("  python main.py test <비디오_파일> [출력_파일]")
        print("\n예시:")
        print("  python main.py webb_logan_09_28")
        print("  python main.py test 001-FC-90.9.mp4")
        sys.exit(1)

    app = PitchAnalysisApp()

    if sys.argv[1] == 'test':
        # 단일 비디오 테스트 모드
        video_path = sys.argv[2]
        output_path = sys.argv[3] if len(sys.argv) > 3 else None

        if not os.path.exists(video_path):
            print(f"오류: 비디오 파일이 존재하지 않습니다: {video_path}")
            sys.exit(1)

        app.run_single_video_test(video_path, output_path)

    else:
        # 전체 분석 모드
        input_dir = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "output_videos"

        csv_path = app.run_analysis(input_dir, output_dir)

        if csv_path:
            print(f"\n분석이 완료되었습니다!")
            print(f"결과 파일: {csv_path}")
        else:
            print(f"\n분석에 실패했습니다.")
            sys.exit(1)


if __name__ == "__main__":
    main()
