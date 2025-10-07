import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from pose_detector import PoseDetector
import mediapipe as mp

class PitcherIdentifier:
    """
    야구 비디오에서 투수를 식별하는 클래스
    """

    def __init__(self, pose_detector: PoseDetector):
        """
        투수 식별기 초기화

        Args:
            pose_detector: 포즈 감지기 인스턴스
        """
        self.pose_detector = pose_detector
        self.mp_pose = mp.solutions.pose

        # 투수 식별을 위한 기준점들
        self.center_region_ratio = 0.3  # 화면 중앙 영역 비율
        self.min_arm_movement_threshold = 50  # 최소 팔 움직임 임계값 (픽셀)
        self.throwing_arm_velocity_threshold = 100  # 투구 팔 속도 임계값

    def identify_pitcher(self, frame: np.ndarray, poses: List[Dict]) -> Optional[Dict]:
        """
        프레임에서 투수를 식별

        Args:
            frame: 입력 프레임
            poses: 감지된 포즈들 리스트

        Returns:
            투수 포즈 데이터 또는 None
        """
        if not poses:
            return None

        frame_height, frame_width = frame.shape[:2]

        # 1단계: 화면 중앙에 있는 사람들 필터링
        center_poses = self._filter_center_poses(poses, frame_width, frame_height)

        if not center_poses:
            return None

        # 2단계: 투구 동작을 보이는 사람들 식별
        throwing_poses = self._identify_throwing_motion(center_poses)

        if not throwing_poses:
            # 중앙에 있는 사람들 중에서 가장 높은 신뢰도를 가진 사람 선택
            return max(center_poses, key=lambda p: p['detection_confidence'])

        # 3단계: 가장 강한 투구 동작을 보이는 사람 선택
        return self._select_primary_thrower(throwing_poses, frame_width, frame_height)

    def _filter_center_poses(self, poses: List[Dict], frame_width: int, frame_height: int) -> List[Dict]:
        """
        화면 중앙 영역에 있는 포즈들을 필터링

        Args:
            poses: 포즈 데이터 리스트
            frame_width: 프레임 너비
            frame_height: 프레임 높이

        Returns:
            중앙 영역의 포즈들
        """
        center_poses = []
        center_x = frame_width // 2
        center_y = frame_height // 2
        center_radius_x = int(frame_width * self.center_region_ratio / 2)
        center_radius_y = int(frame_height * self.center_region_ratio / 2)

        for pose in poses:
            coords = pose['landmark_coords']

            # 어깨나 몸통 중심점으로 중앙 영역 확인
            shoulder_center = self._get_shoulder_center(coords)

            if (center_x - center_radius_x <= shoulder_center[0] <= center_x + center_radius_x and
                center_y - center_radius_y <= shoulder_center[1] <= center_y + center_radius_y):
                center_poses.append(pose)

        return center_poses

    def _get_shoulder_center(self, coords: Dict) -> Tuple[float, float]:
        """
        어깨 중심점 계산

        Args:
            coords: 랜드마크 좌표

        Returns:
            어깨 중심점 (x, y)
        """
        left_shoulder = coords.get('LEFT_SHOULDER')
        right_shoulder = coords.get('RIGHT_SHOULDER')

        if left_shoulder and right_shoulder:
            return ((left_shoulder['x'] + right_shoulder['x']) / 2,
                   (left_shoulder['y'] + right_shoulder['y']) / 2)
        elif left_shoulder:
            return (left_shoulder['x'], left_shoulder['y'])
        elif right_shoulder:
            return (right_shoulder['x'], right_shoulder['y'])
        else:
            # 가슴이나 다른 중심점 사용
            chest = coords.get('LEFT_HIP', coords.get('RIGHT_HIP'))
            return (chest['x'], chest['y']) if chest else (0, 0)

    def _identify_throwing_motion(self, poses: List[Dict]) -> List[Dict]:
        """
        투구 동작을 보이는 포즈들을 식별

        Args:
            poses: 포즈 데이터 리스트

        Returns:
            투구 동작을 보이는 포즈들
        """
        throwing_poses = []

        for pose in poses:
            coords = pose['landmark_coords']

            # 팔 움직임 분석
            arm_movement_score = self._calculate_arm_movement_score(coords)

            if arm_movement_score > self.min_arm_movement_threshold:
                pose['arm_movement_score'] = arm_movement_score
                throwing_poses.append(pose)

        return throwing_poses

    def _calculate_arm_movement_score(self, coords: Dict) -> float:
        """
        팔 움직임 점수 계산 (어깨-팔꿈치-손목 거리 기반)

        Args:
            coords: 랜드마크 좌표

        Returns:
            팔 움직임 점수
        """
        score = 0

        # 오른손 투구 분석
        right_shoulder = coords.get('RIGHT_SHOULDER')
        right_elbow = coords.get('RIGHT_ELBOW')
        right_wrist = coords.get('RIGHT_WRIST')

        if right_shoulder and right_elbow and right_wrist:
            # 어깨-팔꿈치 거리
            shoulder_elbow_dist = self._calculate_distance_2d(right_shoulder, right_elbow)
            # 팔꿈치-손목 거리
            elbow_wrist_dist = self._calculate_distance_2d(right_elbow, right_wrist)

            # 투구 동작에서 팔이 펴지는 정도를 점수로 계산
            if shoulder_elbow_dist > 0 and elbow_wrist_dist > shoulder_elbow_dist * 0.7:
                # 팔이 완전히 펴져 있는 정도를 점수화
                extension_ratio = elbow_wrist_dist / shoulder_elbow_dist
                score += extension_ratio * 100  # 더 자연스러운 점수화

        # 왼손 투구 분석
        left_shoulder = coords.get('LEFT_SHOULDER')
        left_elbow = coords.get('LEFT_ELBOW')
        left_wrist = coords.get('LEFT_WRIST')

        if left_shoulder and left_elbow and left_wrist:
            shoulder_elbow_dist = self._calculate_distance_2d(left_shoulder, left_elbow)
            elbow_wrist_dist = self._calculate_distance_2d(left_elbow, left_wrist)

            if shoulder_elbow_dist > 0 and elbow_wrist_dist > shoulder_elbow_dist * 0.7:
                extension_ratio = elbow_wrist_dist / shoulder_elbow_dist
                score += extension_ratio * 100

        return score

    def _calculate_distance_2d(self, point1: Dict, point2: Dict) -> float:
        """
        두 점 사이의 2D 거리 계산

        Args:
            point1: 첫 번째 점 좌표
            point2: 두 번째 점 좌표

        Returns:
            두 점 사이의 거리
        """
        return np.sqrt((point1['x'] - point2['x'])**2 + (point1['y'] - point2['y'])**2)

    def _calculate_distance(self, point1: Dict, point2: Dict) -> float:
        """
        두 점 사이의 거리 계산

        Args:
            point1: 첫 번째 점 좌표
            point2: 두 번째 점 좌표

        Returns:
            두 점 사이의 거리
        """
        return np.sqrt((point1['x'] - point2['x'])**2 + (point1['y'] - point2['y'])**2)

    def _select_primary_thrower(self, throwing_poses: List[Dict], frame_width: int, frame_height: int) -> Dict:
        """
        여러 투구 동작 중 가장 주요한 투수를 선택

        Args:
            throwing_poses: 투구 동작 포즈들
            frame_width: 프레임 너비
            frame_height: 프레임 높이

        Returns:
            주요 투수 포즈
        """
        if len(throwing_poses) == 1:
            return throwing_poses[0]

        # 가장 높은 팔 움직임 점수를 가진 사람 선택
        return max(throwing_poses, key=lambda p: p.get('arm_movement_score', 0))

    def determine_throwing_arm(self, pose: Dict) -> str:
        """
        투수의 투구 팔을 결정 (왼손/오른손)

        Args:
            pose: 포즈 데이터

        Returns:
            'LEFT' 또는 'RIGHT'
        """
        coords = pose['landmark_coords']

        # 오른손 투구 확인
        right_elbow = coords.get('RIGHT_ELBOW')
        right_wrist = coords.get('RIGHT_WRIST')

        # 왼손 투구 확인
        left_elbow = coords.get('LEFT_ELBOW')
        left_wrist = coords.get('LEFT_WRIST')

        if right_elbow and right_wrist:
            right_arm_extension = self._calculate_distance(right_elbow, right_wrist)
        else:
            right_arm_extension = 0

        if left_elbow and left_wrist:
            left_arm_extension = self._calculate_distance(left_elbow, left_wrist)
        else:
            left_arm_extension = 0

        # 더 많이 펴져 있는 팔을 투구 팔로 결정
        if right_arm_extension > left_arm_extension * 1.2:
            return 'RIGHT'
        elif left_arm_extension > right_arm_extension * 1.2:
            return 'LEFT'
        else:
            # 애매한 경우, 일반적으로 오른손 투구 가정
            return 'RIGHT'
