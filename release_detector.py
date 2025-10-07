import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from elbow_angle_calculator import ElbowAngleCalculator

class ReleaseDetector:
    """
    투구에서 공 릴리스 시점을 식별하는 클래스
    """

    def __init__(self, elbow_calculator: ElbowAngleCalculator):
        """
        릴리스 감지기 초기화

        Args:
            elbow_calculator: 팔꿈치 각도 계산기
        """
        self.elbow_calculator = elbow_calculator

        # 릴리스 감지를 위한 파라미터들
        self.arm_extension_threshold = 0.8  # 팔이 충분히 펴져야 함 (완전 펴짐 80% 이상)
        self.min_angle_threshold = 150  # 최소 팔꿈치 각도 (더 자연스러운 범위)
        self.max_angle_threshold = 180  # 최대 팔꿈치 각도
        self.stability_window = 5  # 안정성 확인을 위한 프레임 윈도우 크기 증가
        self.angle_change_threshold = 5  # 각도 변화 임계값 (안정성 판단용)

    def detect_release_frame(self, pose_sequence: List[Dict], throwing_arm: str,
                           start_frame: int = 0) -> Optional[int]:
        """
        포즈 시퀀스에서 공 릴리스 프레임을 식별

        Args:
            pose_sequence: 포즈 데이터 시퀀스 (시간순)
            throwing_arm: 투구 팔 ('LEFT' 또는 'RIGHT')
            start_frame: 분석 시작 프레임

        Returns:
            릴리스 프레임 인덱스 또는 None
        """
        if len(pose_sequence) < self.stability_window:
            return None

        # 각 프레임의 팔꿈치 각도와 팔 펴짐 정도 계산
        frame_metrics = []
        for i, pose_data in enumerate(pose_sequence):
            if i < start_frame:
                continue

            angle = self.elbow_calculator.calculate_elbow_angle_from_pose(pose_data, throwing_arm)
            extension_ratio = self._calculate_extension_ratio(pose_data, throwing_arm)

            if angle is not None and extension_ratio is not None:
                frame_metrics.append({
                    'frame_idx': i,
                    'angle': angle,
                    'extension_ratio': extension_ratio,
                    'is_arm_extended': extension_ratio >= self.arm_extension_threshold,
                    'angle_in_range': self.min_angle_threshold <= angle <= self.max_angle_threshold
                })

        if not frame_metrics:
            return None

        # 릴리스 후보 프레임들 찾기 (팔이 완전히 펴지고 각도가 최대인 지점)
        release_candidates = self._find_release_candidates(frame_metrics)

        if not release_candidates:
            return None

        # 가장 적합한 릴리스 프레임 선택
        return self._select_best_release_frame(release_candidates, frame_metrics)

    def _calculate_extension_ratio(self, pose_data: Dict, throwing_arm: str) -> Optional[float]:
        """
        포즈 데이터에서 팔 펴짐 비율 계산

        Args:
            pose_data: 포즈 데이터
            throwing_arm: 투구 팔

        Returns:
            팔 펴짐 비율 또는 None
        """
        shoulder, elbow, wrist = self.elbow_calculator.get_arm_landmarks(pose_data, throwing_arm)

        if shoulder and elbow and wrist:
            return self.elbow_calculator.calculate_arm_extension_ratio(shoulder, elbow, wrist)

        return None

    def _find_release_candidates(self, frame_metrics: List[Dict]) -> List[Dict]:
        """
        릴리스 후보 프레임들 찾기

        Args:
            frame_metrics: 프레임 메트릭 데이터들

        Returns:
            릴리스 후보 프레임들
        """
        candidates = []

        for i, metric in enumerate(frame_metrics):
            # 팔이 충분히 펴져 있고 각도가 적절한 범위에 있는 프레임
            if metric['is_arm_extended'] and metric['angle_in_range']:
                # 각도 변화가 안정적인지 확인 (릴리스 시점 주변에서 각도가 급격히 변하지 않음)
                if self._is_stable_angle_change(frame_metrics, i):
                    candidates.append(metric)

        return candidates

    def _is_stable_angle_change(self, frame_metrics: List[Dict], center_idx: int, window: int = 2) -> bool:
        """
        각도 변화가 안정적인지 확인

        Args:
            frame_metrics: 프레임 메트릭 데이터들
            center_idx: 확인할 중심 인덱스
            window: 확인할 주변 프레임 범위

        Returns:
            각도 변화가 안정적인지 여부
        """
        if center_idx < window or center_idx >= len(frame_metrics) - window:
            return False

        center_angle = frame_metrics[center_idx]['angle']
        prev_angles = []
        next_angles = []

        # 이전 프레임들의 각도 수집
        for i in range(max(0, center_idx - window), center_idx):
            if frame_metrics[i]['angle'] is not None:
                prev_angles.append(frame_metrics[i]['angle'])

        # 이후 프레임들의 각도 수집
        for i in range(center_idx + 1, min(len(frame_metrics), center_idx + window + 1)):
            if frame_metrics[i]['angle'] is not None:
                next_angles.append(frame_metrics[i]['angle'])

        if not prev_angles or not next_angles:
            return False

        # 각도 변화율 계산
        prev_change_rate = abs(center_angle - prev_angles[-1]) if prev_angles else 0
        next_change_rate = abs(next_angles[0] - center_angle) if next_angles else 0

        # 변화율이 임계값 이하인지 확인
        return prev_change_rate <= self.angle_change_threshold and next_change_rate <= self.angle_change_threshold

    def _is_local_maximum_angle(self, frame_metrics: List[Dict], center_idx: int,
                               window: int = 2) -> bool:
        """
        주어진 인덱스가 주변에서 각도가 최대인지 확인

        Args:
            frame_metrics: 프레임 메트릭 데이터들
            center_idx: 확인할 중심 인덱스
            window: 확인할 주변 프레임 범위

        Returns:
            로컬 최대점인지 여부
        """
        center_angle = frame_metrics[center_idx]['angle']

        for i in range(max(0, center_idx - window), min(len(frame_metrics), center_idx + window + 1)):
            if i != center_idx and frame_metrics[i]['angle'] > center_angle:
                return False

        return True

    def _select_best_release_frame(self, candidates: List[Dict],
                                 frame_metrics: List[Dict]) -> Optional[int]:
        """
        여러 후보 중에서 최적의 릴리스 프레임을 선택

        Args:
            candidates: 릴리스 후보들
            frame_metrics: 모든 프레임 메트릭

        Returns:
            최적 릴리스 프레임 인덱스 또는 None
        """
        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0]['frame_idx']

        # 가장 높은 각도를 가진 프레임을 선택
        best_candidate = max(candidates, key=lambda c: c['angle'])

        return best_candidate['frame_idx']

    def analyze_throwing_motion(self, pose_sequence: List[Dict], throwing_arm: str) -> Dict:
        """
        투구 동작 전체를 분석하여 릴리스 시점과 관련 메트릭 반환

        Args:
            pose_sequence: 포즈 시퀀스
            throwing_arm: 투구 팔

        Returns:
            분석 결과 딕셔너리
        """
        if len(pose_sequence) < 10:  # 최소 분석 가능한 프레임 수
            return {'release_frame': None, 'error': 'Insufficient frames'}

        release_frame = self.detect_release_frame(pose_sequence, throwing_arm)

        if release_frame is None:
            return {'release_frame': None, 'error': 'No release detected'}

        # 릴리스 시점의 팔꿈치 각도 계산
        release_pose = pose_sequence[release_frame]
        release_angle = self.elbow_calculator.calculate_elbow_angle_from_pose(release_pose, throwing_arm)

        # 릴리스 시점 주변의 각도 변화 분석
        angle_progression = self._analyze_angle_progression(pose_sequence, release_frame, throwing_arm)

        return {
            'release_frame': release_frame,
            'release_angle': release_angle,
            'angle_progression': angle_progression,
            'motion_quality': self._assess_motion_quality(angle_progression)
        }

    def _analyze_angle_progression(self, pose_sequence: List[Dict], release_frame: int,
                                throwing_arm: str, window: int = 10) -> List[Dict]:
        """
        릴리스 시점 주변의 각도 변화를 분석

        Args:
            pose_sequence: 포즈 시퀀스
            release_frame: 릴리스 프레임
            throwing_arm: 투구 팔
            window: 분석할 주변 프레임 범위

        Returns:
            각도 변화 데이터 리스트
        """
        progression = []
        start_frame = max(0, release_frame - window)
        end_frame = min(len(pose_sequence), release_frame + window + 1)

        for i in range(start_frame, end_frame):
            angle = self.elbow_calculator.calculate_elbow_angle_from_pose(pose_sequence[i], throwing_arm)
            extension_ratio = self._calculate_extension_ratio(pose_sequence[i], throwing_arm)

            progression.append({
                'frame_idx': i,
                'relative_frame': i - release_frame,
                'angle': angle,
                'extension_ratio': extension_ratio,
                'is_release_frame': (i == release_frame)
            })

        return progression

    def _assess_motion_quality(self, angle_progression: List[Dict]) -> str:
        """
        투구 동작의 품질을 평가

        Args:
            angle_progression: 각도 변화 데이터

        Returns:
            동작 품질 평가 ('GOOD', 'FAIR', 'POOR')
        """
        if not angle_progression:
            return 'POOR'

        # 릴리스 시점의 각도 확인
        release_angle = None
        for frame in angle_progression:
            if frame['is_release_frame']:
                release_angle = frame['angle']
                break

        if release_angle is None or release_angle < 160:
            return 'POOR'

        # 각도 변화의 부드러움 확인 (급격한 변화가 적은지)
        angles = [f['angle'] for f in angle_progression if f['angle'] is not None]

        if len(angles) < 3:
            return 'FAIR'

        # 각도 변화율의 표준편차 계산
        angle_diffs = [angles[i+1] - angles[i] for i in range(len(angles)-1)]
        if not angle_diffs:
            return 'FAIR'

        std_dev = np.std(angle_diffs)

        if std_dev < 5:  # 변화가 부드러운 경우
            return 'GOOD'
        elif std_dev < 15:
            return 'FAIR'
        else:
            return 'POOR'

    def get_release_frame_alternatives(self, pose_sequence: List[Dict], throwing_arm: str) -> List[int]:
        """
        대안 릴리스 프레임들을 반환 (여러 후보가 있는 경우)

        Args:
            pose_sequence: 포즈 시퀀스
            throwing_arm: 투구 팔

        Returns:
            대안 릴리스 프레임 인덱스 리스트
        """
        if len(pose_sequence) < self.stability_window:
            return []

        frame_metrics = []
        for i, pose_data in enumerate(pose_sequence):
            angle = self.elbow_calculator.calculate_elbow_angle_from_pose(pose_data, throwing_arm)
            extension_ratio = self._calculate_extension_ratio(pose_data, throwing_arm)

            if angle is not None and extension_ratio is not None:
                frame_metrics.append({
                    'frame_idx': i,
                    'angle': angle,
                    'extension_ratio': extension_ratio,
                    'is_arm_extended': extension_ratio >= self.arm_extension_threshold,
                    'angle_in_range': self.min_angle_threshold <= angle <= self.max_angle_threshold
                })

        if not frame_metrics:
            return []

        # 릴리스 조건을 만족하는 모든 프레임 반환
        alternatives = [m['frame_idx'] for m in frame_metrics
                       if m['is_arm_extended'] and m['angle_in_range']]

        return sorted(alternatives)
