import cv2
import numpy as np
import os
from typing import List, Dict, Tuple, Optional
from pose_detector import PoseDetector
from pitcher_identifier import PitcherIdentifier
from elbow_angle_calculator import ElbowAngleCalculator
from release_detector import ReleaseDetector

class VideoProcessor:
    """
    야구 투구 비디오를 처리하고 분석하는 클래스
    """

    def __init__(self, pose_detector: PoseDetector, pitcher_identifier: PitcherIdentifier,
                 elbow_calculator: ElbowAngleCalculator, release_detector: ReleaseDetector):
        """
        비디오 프로세서 초기화

        Args:
            pose_detector: 포즈 감지기
            pitcher_identifier: 투수 식별기
            elbow_calculator: 팔꿈치 각도 계산기
            release_detector: 릴리스 감지기
        """
        self.pose_detector = pose_detector
        self.pitcher_identifier = pitcher_identifier
        self.elbow_calculator = elbow_calculator
        self.release_detector = release_detector

    def process_video(self, video_path: str, output_path: str, show_preview: bool = False) -> Dict:
        """
        비디오 파일을 처리하고 분석

        Args:
            video_path: 입력 비디오 파일 경로
            output_path: 출력 비디오 파일 경로
            show_preview: 미리보기 표시 여부

        Returns:
            분석 결과 딕셔너리
        """
        # 비디오 캡처 열기
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return {'error': f'Cannot open video file: {video_path}'}

        # 비디오 속성 가져오기
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Processing video: {video_path}")
        print(f"Resolution: {frame_width}x{frame_height}, FPS: {fps}, Frames: {total_frames}")

        # 출력 비디오 설정
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        # 분석 데이터 저장을 위한 변수들
        pose_sequence = []
        pitcher_data = None
        throwing_arm = 'RIGHT'  # 기본값
        release_analysis = None

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 50 == 0:  # 진행률 표시
                print(f"Processing frame {frame_count}/{total_frames}")

            # 포즈 감지
            poses = self.pose_detector.detect_poses(frame)

            if poses:
                # 투수 식별
                if pitcher_data is None:
                    pitcher_data = self.pitcher_identifier.identify_pitcher(frame, poses)
                    if pitcher_data:
                        throwing_arm = self.pitcher_identifier.determine_throwing_arm(pitcher_data)
                        print(f"투수 식별 완료 - 투구 팔: {throwing_arm}")

                # 투수가 식별되었으면 해당 포즈만 추적
                if pitcher_data:
                    # 현재 프레임의 투수 포즈 찾기
                    current_pitcher_pose = None
                    for pose in poses:
                        if pose['person_id'] == pitcher_data['person_id']:
                            current_pitcher_pose = pose
                            break

                    if current_pitcher_pose:
                        pose_sequence.append(current_pitcher_pose)

                        # 팔꿈치 각도 계산
                        elbow_angle = self.elbow_calculator.calculate_elbow_angle_from_pose(
                            current_pitcher_pose, throwing_arm
                        )

                        # 프레임에 주석 추가
                        annotated_frame = self._annotate_frame(
                            frame, current_pitcher_pose, elbow_angle, throwing_arm, frame_count
                        )

                        # 릴리스 분석 (시퀀스가 충분히 쌓인 후)
                        if len(pose_sequence) > 30 and release_analysis is None:
                            release_analysis = self.release_detector.analyze_throwing_motion(
                                pose_sequence, throwing_arm
                            )
                            if release_analysis['release_frame'] is not None:
                                print(f"릴리스 프레임 감지: {release_analysis['release_frame']}")
                    else:
                        # 투수가 감지되지 않으면 일반 포즈 감지 결과 사용
                        annotated_frame = self.pose_detector.draw_pose_landmarks(frame, poses[0])
                else:
                    # 투수가 식별되지 않은 경우 첫 번째 포즈 사용
                    annotated_frame = self.pose_detector.draw_pose_landmarks(frame, poses[0])
            else:
                # 포즈가 감지되지 않은 경우 원본 프레임 사용
                annotated_frame = frame

            # 결과 비디오에 쓰기
            out.write(annotated_frame)

            # 미리보기 표시
            if show_preview:
                cv2.imshow('Pitch Analysis', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # 정리
        cap.release()
        out.release()
        if show_preview:
            cv2.destroyAllWindows()

        # 최종 분석 결과 반환
        result = {
            'video_path': video_path,
            'output_path': output_path,
            'total_frames': total_frames,
            'fps': fps,
            'throwing_arm': throwing_arm,
            'release_analysis': release_analysis,
            'frame_count': frame_count
        }

        if release_analysis and release_analysis['release_frame'] is not None:
            release_frame = release_analysis['release_frame']
            if release_frame < len(pose_sequence):
                release_angle = self.elbow_calculator.calculate_elbow_angle_from_pose(
                    pose_sequence[release_frame], throwing_arm
                )
                result['final_elbow_angle'] = release_angle

        return result

    def _annotate_frame(self, frame: np.ndarray, pose_data: Dict, elbow_angle: Optional[float],
                       throwing_arm: str, frame_number: int) -> np.ndarray:
        """
        프레임에 분석 주석을 추가

        Args:
            frame: 입력 프레임
            pose_data: 포즈 데이터
            elbow_angle: 팔꿈치 각도
            throwing_arm: 투구 팔
            frame_number: 프레임 번호

        Returns:
            주석이 추가된 프레임
        """
        # 포즈 랜드마크 그리기
        annotated_frame = self.pose_detector.draw_pose_landmarks(frame, pose_data)

        # 팔꿈치 각도 텍스트 추가
        angle_text = f"Elbow Angle: {elbow_angle:.1f}°" if elbow_angle else "Elbow Angle: N/A"
        annotated_frame = self.pose_detector.draw_text(
            annotated_frame, angle_text, position=(10, 30), color=(0, 255, 0)
        )

        # 투구 팔 정보 추가
        arm_text = f"Throwing Arm: {throwing_arm}"
        annotated_frame = self.pose_detector.draw_text(
            annotated_frame, arm_text, position=(10, 70), color=(255, 255, 0)
        )

        # 프레임 번호 추가
        frame_text = f"Frame: {frame_number}"
        annotated_frame = self.pose_detector.draw_text(
            annotated_frame, frame_text, position=(10, 110), color=(255, 255, 255)
        )

        # 투구 팔의 관절점 연결선 강조 (시각화용)
        self._draw_arm_highlight(annotated_frame, pose_data, throwing_arm)

        return annotated_frame

    def _draw_arm_highlight(self, frame: np.ndarray, pose_data: Dict, throwing_arm: str):
        """
        투구 팔의 관절점을 강조 표시

        Args:
            frame: 입력 프레임
            pose_data: 포즈 데이터
            throwing_arm: 투구 팔
        """
        coords = pose_data['landmark_coords']

        if throwing_arm == 'RIGHT':
            shoulder = coords.get('RIGHT_SHOULDER')
            elbow = coords.get('RIGHT_ELBOW')
            wrist = coords.get('RIGHT_WRIST')
        else:
            shoulder = coords.get('LEFT_SHOULDER')
            elbow = coords.get('LEFT_ELBOW')
            wrist = coords.get('LEFT_WRIST')

        # 어깨-팔꿈치-손목 연결선 그리기 (두꺼운 선으로 강조)
        if shoulder and elbow:
            cv2.line(frame,
                    (int(shoulder['x']), int(shoulder['y'])),
                    (int(elbow['x']), int(elbow['y'])),
                    (0, 0, 255), 3)  # 빨간색 두꺼운 선

        if elbow and wrist:
            cv2.line(frame,
                    (int(elbow['x']), int(elbow['y'])),
                    (int(wrist['x']), int(wrist['y'])),
                    (0, 0, 255), 3)  # 빨간색 두꺼운 선

    def batch_process_videos(self, input_dir: str, output_dir: str, file_pattern: str = "*.mp4") -> List[Dict]:
        """
        디렉토리 내의 모든 비디오 파일을 일괄 처리

        Args:
            input_dir: 입력 디렉토리 경로
            output_dir: 출력 디렉토리 경로
            file_pattern: 파일 패턴

        Returns:
            각 비디오의 처리 결과 리스트
        """
        import glob

        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)

        # 비디오 파일 목록 가져오기
        video_files = glob.glob(os.path.join(input_dir, file_pattern))

        if not video_files:
            print(f"No video files found in {input_dir} with pattern {file_pattern}")
            return []

        print(f"Found {len(video_files)} video files to process")

        results = []
        for video_file in sorted(video_files):
            print(f"\nProcessing: {os.path.basename(video_file)}")

            # 출력 파일 경로 생성
            video_name = os.path.basename(video_file)
            output_file = os.path.join(output_dir, video_name.replace('.mp4', '_processed.mp4'))

            # 비디오 처리
            result = self.process_video(video_file, output_file)

            if 'error' not in result:
                print(f"✓ Successfully processed: {video_name}")
            else:
                print(f"✗ Failed to process {video_name}: {result['error']}")

            results.append(result)

        return results

    def extract_pitch_info_from_filename(self, filename: str) -> Dict:
        """
        파일명에서 투구 정보를 추출

        Args:
            filename: 파일명 (예: "001-FC-90.9.mp4")

        Returns:
            투구 정보 딕셔너리
        """
        try:
            parts = filename.replace('.mp4', '').split('-')
            if len(parts) >= 3:
                return {
                    'pitch_number': int(parts[0]),
                    'pitch_type': parts[1],
                    'velocity_mph': float(parts[2])
                }
        except (ValueError, IndexError):
            pass

        return {
            'pitch_number': None,
            'pitch_type': None,
            'velocity_mph': None
        }
