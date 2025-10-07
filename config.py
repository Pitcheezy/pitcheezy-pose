"""
야구 투구 분석 시스템 설정 파일

이 파일에는 시스템의 모든 설정값이 중앙집중화되어 있습니다.
환경별 설정을 쉽게 관리할 수 있도록 구성되어 있습니다.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any
from pathlib import Path


@dataclass
class MediaPipeConfig:
    """MediaPipe 관련 설정"""
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    model_complexity: int = 1  # 0, 1, 2 중 선택


@dataclass
class PitcherIdentifierConfig:
    """투수 식별 관련 설정"""
    center_region_ratio: float = 0.3  # 화면 중앙 영역 비율
    min_arm_movement_threshold: float = 50.0  # 최소 팔 움직임 임계값 (픽셀)
    throwing_arm_velocity_threshold: float = 100.0  # 투구 팔 속도 임계값


@dataclass
class ReleaseDetectorConfig:
    """릴리스 감지 관련 설정"""
    arm_extension_threshold: float = 0.8  # 팔 펴짐 임계값 (완전 펴짐 80% 이상)
    min_angle_threshold: float = 150.0  # 최소 팔꿈치 각도 (도)
    max_angle_threshold: float = 180.0  # 최대 팔꿈치 각도 (도)
    stability_window: int = 5  # 안정성 확인 윈도우 크기
    angle_change_threshold: float = 5.0  # 각도 변화 임계값


@dataclass
class VideoProcessorConfig:
    """비디오 처리 관련 설정"""
    frame_skip_interval: int = 50  # 진행률 표시 간격
    min_frames_for_analysis: int = 30  # 분석을 위한 최소 프레임 수
    output_video_quality: int = 85  # 출력 비디오 품질 (1-100)


@dataclass
class AnalysisConfig:
    """분석 관련 설정"""
    mediapipe: MediaPipeConfig = field(default_factory=MediaPipeConfig)
    pitcher_identifier: PitcherIdentifierConfig = field(default_factory=PitcherIdentifierConfig)
    release_detector: ReleaseDetectorConfig = field(default_factory=ReleaseDetectorConfig)
    video_processor: VideoProcessorConfig = field(default_factory=VideoProcessorConfig)

    # 파일 경로 설정
    default_input_dir: str = "webb_logan_09_28"
    default_output_dir: str = "output_videos"

    # 처리 옵션
    show_preview: bool = False
    save_intermediate_results: bool = False

    # 로깅 설정
    log_level: str = "INFO"
    log_file: str = "logs/pitch_analysis.log"


class ConfigManager:
    """설정 관리 클래스"""

    def __init__(self, environment: str = "development"):
        """
        설정 관리자 초기화

        Args:
            environment: 실행 환경 ("development", "production", "testing")
        """
        self.environment = environment
        self.config = self._load_config()

    def _load_config(self) -> AnalysisConfig:
        """환경에 따른 설정 로드"""
        base_config = AnalysisConfig()

        # 환경별 설정 오버라이드
        if self.environment == "production":
            # 운영 환경 설정
            base_config.log_level = "WARNING"
            base_config.show_preview = False
            base_config.mediapipe.min_detection_confidence = 0.7
            base_config.mediapipe.min_tracking_confidence = 0.7

        elif self.environment == "testing":
            # 테스트 환경 설정
            base_config.log_level = "DEBUG"
            base_config.show_preview = False
            base_config.save_intermediate_results = True

        # 환경 변수로부터 설정 오버라이드
        self._override_from_env(base_config)

        return base_config

    def _override_from_env(self, config: AnalysisConfig):
        """환경 변수로부터 설정 오버라이드"""
        # MediaPipe 설정
        if os.getenv("MEDIAPIPE_DETECTION_CONFIDENCE"):
            config.mediapipe.min_detection_confidence = float(os.getenv("MEDIAPIPE_DETECTION_CONFIDENCE"))

        if os.getenv("MEDIAPIPE_TRACKING_CONFIDENCE"):
            config.mediapipe.min_tracking_confidence = float(os.getenv("MEDIAPIPE_TRACKING_CONFIDENCE"))

        # 입력/출력 디렉토리 설정
        if os.getenv("INPUT_DIR"):
            config.default_input_dir = os.getenv("INPUT_DIR")

        if os.getenv("OUTPUT_DIR"):
            config.default_output_dir = os.getenv("OUTPUT_DIR")

        # 처리 옵션
        if os.getenv("SHOW_PREVIEW"):
            config.show_preview = os.getenv("SHOW_PREVIEW").lower() in ("true", "1", "yes")

        # 로깅 설정
        if os.getenv("LOG_LEVEL"):
            config.log_level = os.getenv("LOG_LEVEL")

        if os.getenv("LOG_FILE"):
            config.log_file = os.getenv("LOG_FILE")

    def get_config(self) -> AnalysisConfig:
        """현재 설정 반환"""
        return self.config

    def update_config(self, **kwargs):
        """설정 업데이트"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                # 중첩된 설정 업데이트 (예: mediapipe.min_detection_confidence)
                self._update_nested_config(key, value)

    def _update_nested_config(self, key_path: str, value: Any):
        """중첩된 설정 업데이트"""
        keys = key_path.split('.')
        current = self.config

        # 마지막 키를 제외한 모든 중간 객체 탐색
        for key in keys[:-1]:
            if hasattr(current, key):
                current = getattr(current, key)
            else:
                return  # 경로가 존재하지 않음

        # 마지막 키 설정
        if hasattr(current, keys[-1]):
            setattr(current, keys[-1], value)

    def save_config(self, file_path: str = "config.yaml"):
        """현재 설정을 파일로 저장"""
        import yaml

        # 설정을 딕셔너리로 변환
        config_dict = self._config_to_dict(self.config)

        # YAML 파일로 저장
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)

        print(f"설정이 {file_path}에 저장되었습니다.")

    def _config_to_dict(self, obj: Any) -> Dict:
        """객체를 딕셔너리로 변환"""
        if hasattr(obj, '__dataclass_fields__'):
            # 데이터클래스인 경우
            result = {}
            for field_name, field_def in obj.__dataclass_fields__.items():
                value = getattr(obj, field_name)
                if hasattr(value, '__dataclass_fields__'):
                    # 중첩된 데이터클래스인 경우 재귀 호출
                    result[field_name] = self._config_to_dict(value)
                else:
                    result[field_name] = value
            return result
        else:
            return obj


# 전역 설정 인스턴스
_config_manager: ConfigManager = None


def get_config(environment: str = None) -> AnalysisConfig:
    """
    전역 설정 인스턴스를 반환

    Args:
        environment: 실행 환경 (지정하지 않으면 자동 감지)

    Returns:
        설정 객체
    """
    global _config_manager

    if _config_manager is None:
        if environment is None:
            environment = os.getenv("ENVIRONMENT", "development")
        _config_manager = ConfigManager(environment)

    return _config_manager.get_config()


def init_config(environment: str = "development") -> ConfigManager:
    """
    설정 관리자를 초기화

    Args:
        environment: 실행 환경

    Returns:
        설정 관리자 인스턴스
    """
    global _config_manager
    _config_manager = ConfigManager(environment)
    return _config_manager


# 기본 설정 인스턴스 생성
default_config = get_config()
