import mediapipe as mp
import cv2
import numpy as np
from typing import List, Tuple, Optional

class PoseDetector:
    """
    MediaPipe를 사용한 포즈 감지 클래스
    """

    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        """
        포즈 감지기 초기화

        Args:
            min_detection_confidence: 최소 감지 신뢰도
            min_tracking_confidence: 최소 추적 신뢰도
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def detect_poses(self, frame: np.ndarray) -> List[dict]:
        """
        프레임에서 모든 포즈를 감지

        Args:
            frame: 입력 프레임 (RGB 형식)

        Returns:
            감지된 포즈들의 리스트 (각 포즈는 랜드마크와 신뢰도 정보 포함)
        """
        # BGR을 RGB로 변환
        if frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame

        # 포즈 감지 수행
        results = self.pose.process(frame_rgb)

        poses = []
        if results.pose_landmarks:
            # MediaPipe 결과 구조 확인 및 처리
            landmarks_list = results.pose_landmarks

            # 단일 사람의 경우
            if hasattr(landmarks_list, 'landmark'):
                # 단일 포즈 처리
                pose_data = {
                    'person_id': 0,
                    'landmarks': landmarks_list,
                    'landmark_coords': self._extract_landmark_coordinates(landmarks_list, frame.shape),
                    'detection_confidence': self._calculate_overall_confidence(landmarks_list)
                }
                poses.append(pose_data)
            else:
                # 여러 사람의 포즈 처리 (리스트인 경우)
                try:
                    for person_idx in range(len(landmarks_list)):
                        landmarks = landmarks_list[person_idx]
                        pose_data = {
                            'person_id': person_idx,
                            'landmarks': landmarks,
                            'landmark_coords': self._extract_landmark_coordinates(landmarks, frame.shape),
                            'detection_confidence': self._calculate_overall_confidence(landmarks)
                        }
                        poses.append(pose_data)
                except (TypeError, AttributeError):
                    # 예상치 못한 형태인 경우 단일 포즈로 처리
                    pose_data = {
                        'person_id': 0,
                        'landmarks': landmarks_list,
                        'landmark_coords': self._extract_landmark_coordinates(landmarks_list, frame.shape),
                        'detection_confidence': self._calculate_overall_confidence(landmarks_list)
                    }
                    poses.append(pose_data)

        return poses

    def _extract_landmark_coordinates(self, landmarks, frame_shape: Tuple[int, int, int]) -> dict:
        """
        랜드마크 좌표 추출 (픽셀 좌표와 정규화 좌표)

        Args:
            landmarks: MediaPipe 포즈 랜드마크
            frame_shape: 프레임 형태 (높이, 너비, 채널)

        Returns:
            랜드마크 좌표 딕셔너리
        """
        h, w = frame_shape[:2]
        coords = {}

        for landmark in self.mp_pose.PoseLandmark:
            idx = landmark.value
            if idx < len(landmarks.landmark):
                lm = landmarks.landmark[idx]
                coords[landmark.name] = {
                    'x': lm.x * w,  # 픽셀 x 좌표
                    'y': lm.y * h,  # 픽셀 y 좌표
                    'z': lm.z,      # 깊이 좌표 (정규화)
                    'visibility': lm.visibility
                }

        return coords

    def _calculate_overall_confidence(self, landmarks) -> float:
        """
        전체 포즈 감지의 신뢰도 계산

        Args:
            landmarks: MediaPipe 포즈 랜드마크

        Returns:
            평균 신뢰도
        """
        total_confidence = 0
        valid_landmarks = 0

        for landmark in landmarks.landmark:
            if landmark.visibility > 0:
                total_confidence += landmark.visibility
                valid_landmarks += 1

        return total_confidence / valid_landmarks if valid_landmarks > 0 else 0

    def draw_pose_landmarks(self, frame: np.ndarray, pose_data: dict, connections: bool = True) -> np.ndarray:
        """
        프레임에 포즈 랜드마크와 연결선을 그림

        Args:
            frame: 입력 프레임
            pose_data: 포즈 데이터
            connections: 연결선 표시 여부

        Returns:
            주석이 추가된 프레임
        """
        annotated_frame = frame.copy()

        if connections:
            # 포즈 연결선 그리기
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                pose_data['landmarks'],
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
        else:
            # 랜드마크만 그리기
            for landmark in pose_data['landmarks'].landmark:
                if landmark.visibility > 0.5:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(annotated_frame, (x, y), 3, (0, 255, 0), -1)

        return annotated_frame

    def draw_text(self, frame: np.ndarray, text: str, position: Tuple[int, int] = (10, 30),
                  font_scale: float = 0.7, color: Tuple[int, int, int] = (255, 255, 255),
                  thickness: int = 2, bg_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
        """
        프레임에 텍스트를 그림

        Args:
            frame: 입력 프레임
            text: 표시할 텍스트
            position: 텍스트 위치
            font_scale: 폰트 크기
            color: 텍스트 색상
            thickness: 텍스트 두께
            bg_color: 배경 색상

        Returns:
            텍스트가 추가된 프레임
        """
        annotated_frame = frame.copy()

        # 배경 사각형 그리기
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(annotated_frame,
                     (position[0], position[1] - text_height - 5),
                     (position[0] + text_width + 5, position[1] + 5),
                     bg_color, -1)

        # 텍스트 그리기
        cv2.putText(annotated_frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale, color, thickness, cv2.LINE_AA)

        return annotated_frame
