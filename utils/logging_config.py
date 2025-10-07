"""
로깅 설정 모듈

이 모듈은 프로젝트 전체에 걸쳐 일관된 로깅을 제공합니다.
다양한 환경(개발/운영/테스트)에 맞는 로깅 설정을 지원합니다.
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional


class PitchAnalysisFormatter(logging.Formatter):
    """투구 분석 시스템 전용 로그 포맷터"""

    # 색상 코드 정의 (터미널 출력용)
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }

    def __init__(self, use_colors: bool = True, show_timestamp: bool = True):
        """
        포맷터 초기화

        Args:
            use_colors: 색상 사용 여부 (터미널 출력 시)
            show_timestamp: 타임스탬프 표시 여부
        """
        self.use_colors = use_colors and sys.stdout.isatty()
        self.show_timestamp = show_timestamp

        # 로그 포맷 정의
        if show_timestamp:
            fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        else:
            fmt = '%(name)s - %(levelname)s - %(message)s'

        super().__init__(fmt, datefmt='%Y-%m-%d %H:%M:%S')

    def format(self, record: logging.LogRecord) -> str:
        """로그 레코드 포맷팅"""
        # 색상 적용
        if self.use_colors and record.levelname in self.COLORS:
            color = self.COLORS[record.levelname]
            reset = self.COLORS['RESET']
            original_levelname = record.levelname
            record.levelname = f"{color}{record.levelname}{reset}"

            # 메시지도 색상 적용 (특정 키워드에 대해)
            if '오류' in record.getMessage() or '실패' in record.getMessage():
                record.msg = f"{color}{record.msg}{reset}"

            formatted = super().format(record)
            record.levelname = original_levelname  # 원래 값 복원
            return formatted
        else:
            return super().format(record)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    use_colors: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    environment: str = "development"
) -> logging.Logger:
    """
    로깅 시스템 설정

    Args:
        level: 로그 레벨 ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        log_file: 로그 파일 경로 (None이면 파일 로깅 비활성화)
        use_colors: 색상 사용 여부
        max_file_size: 로그 파일 최대 크기 (바이트)
        backup_count: 로그 파일 백업 개수
        environment: 실행 환경

    Returns:
        설정된 로거 인스턴스
    """
    # 루트 로거 설정
    logger = logging.getLogger('pitch_analysis')
    logger.setLevel(getattr(logging, level.upper()))

    # 기존 핸들러 제거 (재설정 방지)
    logger.handlers.clear()

    # 포맷터 생성
    formatter = PitchAnalysisFormatter(use_colors=use_colors)

    # 콘솔 핸들러 (항상 활성화)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # 환경별 로그 레벨 조정
    if environment == "production":
        console_handler.setLevel(logging.WARNING)
    elif environment == "testing":
        console_handler.setLevel(logging.DEBUG)
    else:
        console_handler.setLevel(getattr(logging, level.upper()))

    logger.addHandler(console_handler)

    # 파일 로깅 설정 (log_file이 지정된 경우)
    if log_file:
        # 로그 디렉토리 생성
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # RotatingFileHandler 설정 (파일 크기 제한)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )

        # 파일용 포맷터 (색상 없이)
        file_formatter = PitchAnalysisFormatter(use_colors=False, show_timestamp=True)
        file_handler.setFormatter(file_formatter)

        # 파일 로그 레벨 설정
        file_handler.setLevel(getattr(logging, level.upper()))
        logger.addHandler(file_handler)

        print(f"로그 파일이 설정되었습니다: {log_file}")

    # 서드파티 라이브러리 로그 억제 (선택사항)
    if environment == "production":
        logging.getLogger('mediapipe').setLevel(logging.WARNING)
        logging.getLogger('opencv').setLevel(logging.WARNING)

    return logger


def get_logger(name: str, config_level: str = None) -> logging.Logger:
    """
    이름으로 로거를 가져옴

    Args:
        name: 로거 이름 (보통 모듈명)
        config_level: 설정에서 읽은 로그 레벨 (기본값 사용시 None)

    Returns:
        설정된 로거 인스턴스
    """
    logger = logging.getLogger(f'pitch_analysis.{name}')

    # 설정 레벨이 지정된 경우 적용
    if config_level:
        logger.setLevel(getattr(logging, config_level.upper()))

    return logger


# 편의 함수들
def log_function_call(func):
    """함수 호출을 로깅하는 데코레이터"""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"함수 호출: {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"함수 완료: {func.__name__}")
            return result
        except Exception as e:
            logger.error(f"함수 오류: {func.__name__} - {str(e)}")
            raise
    return wrapper


def log_performance(func):
    """함수 실행 시간을 로깅하는 데코레이터"""
    import time

    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"함수 실행 시간: {func.__name__} - {execution_time:.3f}초")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"함수 실행 실패: {func.__name__} - {execution_time:.3f}초 - {str(e)}")
            raise

    return wrapper
