"""
팔꿈치 각도 계산기 테스트

이 모듈은 ElbowAngleCalculator 클래스의 기능을 테스트합니다.
"""

import unittest
import numpy as np
from pitch_analysis.analysis.elbow_angle_calculator import ElbowAngleCalculator


class TestElbowAngleCalculator(unittest.TestCase):
    """팔꿈치 각도 계산기 테스트 클래스"""

    def setUp(self):
        """테스트 전에 실행되는 설정"""
        self.calculator = ElbowAngleCalculator()

    def test_calculate_elbow_angle_valid_points(self):
        """유효한 점들로 팔꿈치 각도 계산 테스트"""
        # 테스트용 좌표 (간단한 케이스)
        shoulder = {'x': 0, 'y': 0, 'z': 0}
        elbow = {'x': 1, 'y': 0, 'z': 0}
        wrist = {'x': 2, 'y': 0, 'z': 0}

        angle = self.calculator.calculate_elbow_angle(shoulder, elbow, wrist)

        # 완전히 펴진 팔의 각도는 180도에 가까워야 함
        self.assertIsNotNone(angle)
        self.assertGreaterEqual(angle, 179.0)  # 180도에 매우 가까움
        self.assertLessEqual(angle, 181.0)

    def test_calculate_elbow_angle_bent_arm(self):
        """구부러진 팔의 각도 계산 테스트"""
        # 구부러진 팔 케이스 (약 90도)
        shoulder = {'x': 0, 'y': 0, 'z': 0}
        elbow = {'x': 1, 'y': 0, 'z': 0}
        wrist = {'x': 1, 'y': 1, 'z': 0}  # 아래쪽으로 90도 구부림

        angle = self.calculator.calculate_elbow_angle(shoulder, elbow, wrist)

        self.assertIsNotNone(angle)
        self.assertGreaterEqual(angle, 89.0)  # 약 90도
        self.assertLessEqual(angle, 91.0)

    def test_calculate_elbow_angle_invalid_points(self):
        """유효하지 않은 점들로 계산 시도 테스트"""
        # 동일한 점들 (거리 0)
        shoulder = {'x': 0, 'y': 0, 'z': 0}
        elbow = {'x': 0, 'y': 0, 'z': 0}
        wrist = {'x': 0, 'y': 0, 'z': 0}

        angle = self.calculator.calculate_elbow_angle(shoulder, elbow, wrist)

        # 거리가 0이므로 계산 불가능
        self.assertIsNone(angle)

    def test_calculate_elbow_angle_from_pose_right_arm(self):
        """포즈 데이터에서 오른손 팔꿈치 각도 계산 테스트"""
        # 모의 포즈 데이터
        pose_data = {
            'landmark_coords': {
                'RIGHT_SHOULDER': {'x': 0.5, 'y': 0.3, 'z': 0.1},
                'RIGHT_ELBOW': {'x': 0.6, 'y': 0.4, 'z': 0.1},
                'RIGHT_WRIST': {'x': 0.7, 'y': 0.5, 'z': 0.1}
            }
        }

        angle = self.calculator.calculate_elbow_angle_from_pose(pose_data, 'RIGHT')

        self.assertIsNotNone(angle)
        self.assertIsInstance(angle, float)

    def test_calculate_elbow_angle_from_pose_left_arm(self):
        """포즈 데이터에서 왼손 팔꿈치 각도 계산 테스트"""
        pose_data = {
            'landmark_coords': {
                'LEFT_SHOULDER': {'x': 0.5, 'y': 0.3, 'z': 0.1},
                'LEFT_ELBOW': {'x': 0.4, 'y': 0.4, 'z': 0.1},
                'LEFT_WRIST': {'x': 0.3, 'y': 0.5, 'z': 0.1}
            }
        }

        angle = self.calculator.calculate_elbow_angle_from_pose(pose_data, 'LEFT')

        self.assertIsNotNone(angle)
        self.assertIsInstance(angle, float)

    def test_calculate_elbow_angle_from_pose_missing_landmarks(self):
        """랜드마크가 없는 경우 테스트"""
        pose_data = {
            'landmark_coords': {
                'RIGHT_SHOULDER': {'x': 0.5, 'y': 0.3, 'z': 0.1}
                # 팔꿈치와 손목 없음
            }
        }

        angle = self.calculator.calculate_elbow_angle_from_pose(pose_data, 'RIGHT')

        # 필수 랜드마크가 없으므로 None 반환
        self.assertIsNone(angle)

    def test_is_arm_extended(self):
        """팔이 충분히 펴져 있는지 확인 테스트"""
        # 펴진 팔 케이스
        pose_data = {
            'landmark_coords': {
                'RIGHT_SHOULDER': {'x': 0, 'y': 0, 'z': 0},
                'RIGHT_ELBOW': {'x': 1, 'y': 0, 'z': 0},
                'RIGHT_WRIST': {'x': 1.9, 'y': 0, 'z': 0}  # 어깨-팔꿈치 거리의 90%
            }
        }

        is_extended = self.calculator.is_arm_extended(pose_data, 'RIGHT', threshold=0.8)

        # 기본 임계값 0.8 이상이므로 True여야 함
        self.assertTrue(is_extended)

    def test_get_arm_coordinates_2d(self):
        """2D 좌표 반환 테스트"""
        pose_data = {
            'landmark_coords': {
                'RIGHT_SHOULDER': {'x': 0.5, 'y': 0.3, 'z': 0.1},
                'RIGHT_ELBOW': {'x': 0.6, 'y': 0.4, 'z': 0.1},
                'RIGHT_WRIST': {'x': 0.7, 'y': 0.5, 'z': 0.1}
            }
        }

        shoulder, elbow, wrist = self.calculator.get_arm_coordinates_2d(pose_data, 'RIGHT')

        self.assertIsNotNone(shoulder)
        self.assertIsNotNone(elbow)
        self.assertIsNotNone(wrist)
        self.assertEqual(shoulder, (0.5, 0.3))
        self.assertEqual(elbow, (0.6, 0.4))
        self.assertEqual(wrist, (0.7, 0.5))


if __name__ == '__main__':
    unittest.main()
