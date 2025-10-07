import numpy as np
from typing import Dict, Tuple, Optional

class ElbowAngleCalculator:
    """
    팔꿈치 각도를 계산하는 클래스
    """

    def __init__(self):
        """
        팔꿈치 각도 계산기 초기화
        """
        pass

    def calculate_elbow_angle(self, shoulder: Dict, elbow: Dict, wrist: Dict) -> Optional[float]:
        """
        어깨-팔꿈치-손목 세 점을 사용하여 팔꿈치 각도 계산

        Args:
            shoulder: 어깨 좌표 (x, y, z)
            elbow: 팔꿈치 좌표 (x, y, z)
            wrist: 손목 좌표 (x, y, z)

        Returns:
            팔꿈치 각도 (도 단위) 또는 None (계산 불가능한 경우)
        """
        try:
            # 2D 평면상에서 계산 (깊이 정보는 무시하고 x,y 좌표만 사용)
            shoulder_2d = np.array([shoulder['x'], shoulder['y']])
            elbow_2d = np.array([elbow['x'], elbow['y']])
            wrist_2d = np.array([wrist['x'], wrist['y']])

            # 벡터 계산
            vector_shoulder_to_elbow = elbow_2d - shoulder_2d
            vector_elbow_to_wrist = wrist_2d - elbow_2d

            # 벡터 정규화
            norm_se = np.linalg.norm(vector_shoulder_to_elbow)
            norm_ew = np.linalg.norm(vector_elbow_to_wrist)

            if norm_se == 0 or norm_ew == 0:
                return None

            vector_se_normalized = vector_shoulder_to_elbow / norm_se
            vector_ew_normalized = vector_elbow_to_wrist / norm_ew

            # 코사인 각도 계산 (내적 사용)
            dot_product = np.dot(vector_se_normalized, vector_ew_normalized)
            dot_product = np.clip(dot_product, -1.0, 1.0)  # 수치적 안정성을 위해 클리핑

            angle_radians = np.arccos(dot_product)
            angle_degrees = np.degrees(angle_radians)

            return round(angle_degrees, 2)

        except (KeyError, TypeError, ZeroDivisionError):
            return None

    def calculate_elbow_angle_from_pose(self, pose_data: Dict, throwing_arm: str) -> Optional[float]:
        """
        포즈 데이터에서 팔꿈치 각도 계산

        Args:
            pose_data: 포즈 데이터 (landmark_coords 포함)
            throwing_arm: 투구 팔 ('LEFT' 또는 'RIGHT')

        Returns:
            팔꿈치 각도 (도 단위) 또는 None
        """
        coords = pose_data['landmark_coords']

        if throwing_arm == 'RIGHT':
            shoulder = coords.get('RIGHT_SHOULDER')
            elbow = coords.get('RIGHT_ELBOW')
            wrist = coords.get('RIGHT_WRIST')
        elif throwing_arm == 'LEFT':
            shoulder = coords.get('LEFT_SHOULDER')
            elbow = coords.get('LEFT_ELBOW')
            wrist = coords.get('LEFT_WRIST')
        else:
            return None

        if shoulder and elbow and wrist:
            return self.calculate_elbow_angle(shoulder, elbow, wrist)

        return None

    def get_arm_landmarks(self, pose_data: Dict, throwing_arm: str) -> Tuple[Optional[Dict], Optional[Dict], Optional[Dict]]:
        """
        투구 팔의 랜드마크 좌표 반환

        Args:
            pose_data: 포즈 데이터
            throwing_arm: 투구 팔

        Returns:
            (어깨, 팔꿈치, 손목) 좌표 튜플
        """
        coords = pose_data['landmark_coords']

        if throwing_arm == 'RIGHT':
            return (
                coords.get('RIGHT_SHOULDER'),
                coords.get('RIGHT_ELBOW'),
                coords.get('RIGHT_WRIST')
            )
        elif throwing_arm == 'LEFT':
            return (
                coords.get('LEFT_SHOULDER'),
                coords.get('LEFT_ELBOW'),
                coords.get('LEFT_WRIST')
            )

        return None, None, None

    def calculate_arm_extension_ratio(self, shoulder: Dict, elbow: Dict, wrist: Dict) -> Optional[float]:
        """
        팔의 펴짐 정도를 계산 (어깨-팔꿈치 거리 대비 팔꿈치-손목 거리)

        Args:
            shoulder: 어깨 좌표
            elbow: 팔꿈치 좌표
            wrist: 손목 좌표

        Returns:
            팔 펴짐 비율 또는 None
        """
        try:
            shoulder_elbow_dist = np.linalg.norm(
                np.array([shoulder['x'], shoulder['y'], shoulder['z']]) -
                np.array([elbow['x'], elbow['y'], elbow['z']])
            )
            elbow_wrist_dist = np.linalg.norm(
                np.array([elbow['x'], elbow['y'], elbow['z']]) -
                np.array([wrist['x'], wrist['y'], wrist['z']])
            )

            if shoulder_elbow_dist == 0:
                return None

            return elbow_wrist_dist / shoulder_elbow_dist

        except (KeyError, TypeError):
            return None

    def is_arm_extended(self, pose_data: Dict, throwing_arm: str, threshold: float = 0.8) -> bool:
        """
        팔이 충분히 펴져 있는지 확인

        Args:
            pose_data: 포즈 데이터
            throwing_arm: 투구 팔
            threshold: 펴짐 임계값

        Returns:
            팔이 충분히 펴져 있는지 여부
        """
        shoulder, elbow, wrist = self.get_arm_landmarks(pose_data, throwing_arm)

        if shoulder and elbow and wrist:
            extension_ratio = self.calculate_arm_extension_ratio(shoulder, elbow, wrist)
            return extension_ratio is not None and extension_ratio >= threshold

        return False

    def get_arm_coordinates_2d(self, pose_data: Dict, throwing_arm: str) -> Tuple[Optional[Tuple], Optional[Tuple], Optional[Tuple]]:
        """
        2D 좌표만 반환 (시각화를 위해)

        Args:
            pose_data: 포즈 데이터
            throwing_arm: 투구 팔

        Returns:
            (어깨, 팔꿈치, 손목) 2D 좌표 튜플
        """
        coords = pose_data['landmark_coords']

        if throwing_arm == 'RIGHT':
            shoulder = coords.get('RIGHT_SHOULDER')
            elbow = coords.get('RIGHT_ELBOW')
            wrist = coords.get('RIGHT_WRIST')
        elif throwing_arm == 'LEFT':
            shoulder = coords.get('LEFT_SHOULDER')
            elbow = coords.get('LEFT_ELBOW')
            wrist = coords.get('LEFT_WRIST')
        else:
            return None, None, None

        def extract_2d_coords(point):
            return (point['x'], point['y']) if point else None

        return (
            extract_2d_coords(shoulder),
            extract_2d_coords(elbow),
            extract_2d_coords(wrist)
        )
