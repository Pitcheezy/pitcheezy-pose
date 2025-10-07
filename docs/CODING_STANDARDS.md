# 야구 투구 분석 시스템 코딩 표준

이 문서는 팀 프로젝트에서 일관된 코드 스타일과 품질을 유지하기 위한 가이드라인입니다.

## 📋 목차

1. [코드 스타일](#코드-스타일)
2. [네이밍 규칙](#네이밍-규칙)
3. [문서화 표준](#문서화-표준)
4. [오류 처리](#오류-처리)
5. [테스트 작성](#테스트-작성)
6. [커밋 메시지 규칙](#커밋-메시지-규칙)

## 코드 스타일

### Python 스타일 가이드
- **기본 규칙**: PEP 8을 따릅니다.
- **라인 길이**: 88자 (Black 포맷터 권장)
- **들여쓰기**: 공백 4칸 사용

### 코드 포맷팅
```bash
# 프로젝트 루트에서 실행
black src/ tests/
isort src/ tests/  # import 정렬
flake8 src/ tests/  # 린팅
```

### 예시
```python
# 좋은 예시
class BaseballAnalyzer:
    """야구 분석기 클래스"""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = get_logger(__name__)

    def analyze_pitch(self, video_path: str) -> Dict[str, Any]:
        """투구 비디오를 분석합니다."""
        try:
            result = self._process_video(video_path)
            return result
        except FileNotFoundError as e:
            self.logger.error(f"비디오 파일을 찾을 수 없습니다: {video_path}")
            raise

# 나쁜 예시 (피해야 할 코드)
class badClass:
    def __init__(self,c):
        self.c=c

    def func(self,x):
        return x*2
```

## 네이밍 규칙

### 클래스
- **PascalCase** 사용: `BaseballAnalyzer`, `PitchIdentifier`
- **의미 있는 이름** 사용: 동사나 명사 형태로 명확히

### 함수와 메서드
- **snake_case** 사용: `analyze_video()`, `calculate_elbow_angle()`
- **동사 형태**로 시작: `get_`, `set_`, `calculate_`, `process_` 등

### 변수
- **snake_case** 사용: `frame_count`, `elbow_angle`
- **의미 있는 이름** 사용: 축약어 지양

### 상수
- **SCREAMING_SNAKE_CASE** 사용: `MAX_RETRY_COUNT = 3`

### 모듈과 패키지
- **snake_case** 사용: `video_processor.py`, `elbow_calculator.py`
- **기능별 그룹화**: `analysis/`, `utils/`, `core/`

## 문서화 표준

### 함수와 메서드
모든 public 함수와 메서드는 독스트링을 가져야 합니다.

```python
def calculate_elbow_angle(
    shoulder: Dict[str, float],
    elbow: Dict[str, float],
    wrist: Dict[str, float]
) -> Optional[float]:
    """
    어깨-팔꿈치-손목 세 점을 사용하여 팔꿈치 각도를 계산합니다.

    Args:
        shoulder: 어깨 좌표 (x, y, z 키 포함)
        elbow: 팔꿈치 좌표 (x, y, z 키 포함)
        wrist: 손목 좌표 (x, y, z 키 포함)

    Returns:
        팔꿈치 각도 (도 단위) 또는 계산 불가능한 경우 None

    Raises:
        ValueError: 입력 좌표가 유효하지 않은 경우

    Example:
        >>> calc = ElbowAngleCalculator()
        >>> angle = calc.calculate_elbow_angle(
        ...     {'x': 0, 'y': 0, 'z': 0},
        ...     {'x': 1, 'y': 0, 'z': 0},
        ...     {'x': 2, 'y': 0, 'z': 0}
        ... )
        >>> print(angle)
        180.0
    """
```

### 클래스
클래스도 독스트링을 가져야 합니다.

```python
class ElbowAngleCalculator:
    """
    야구 투구 비디오에서 팔꿈치 각도를 계산하는 클래스입니다.

    이 클래스는 MediaPipe 포즈 감지 결과를 바탕으로 투수의
    팔꿈치 각도를 3차원 공간에서 계산합니다.

    Attributes:
        config (Dict): 계산 설정값들
    """
```

### 모듈
모듈 상단에 목적과 주요 클래스를 설명합니다.

```python
"""
팔꿈치 각도 계산 모듈

이 모듈은 야구 투구 분석에서 팔꿈치 각도를 계산하는 기능을 제공합니다.

주요 클래스:
    - ElbowAngleCalculator: 팔꿈치 각도 계산기
"""
```

## 오류 처리

### 예외 처리 원칙
1. **특정 예외**를 잡아야 합니다 (너무 광범위한 except 사용 금지)
2. **로깅**과 함께 예외를 다시 발생시킵니다
3. **사용자 친화적**인 오류 메시지를 제공합니다

### 좋은 예시
```python
def process_video(self, video_path: str) -> Dict[str, Any]:
    """비디오 파일을 처리합니다."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"비디오 파일을 열 수 없습니다: {video_path}")

        # 처리 로직...

    except FileNotFoundError as e:
        self.logger.error(f"비디오 파일 오류: {e}")
        raise VideoProcessingError(f"비디오 처리 실패: {video_path}") from e
    except cv2.error as e:
        self.logger.error(f"OpenCV 오류: {e}")
        raise VideoProcessingError("비디오 디코딩 오류") from e
    except Exception as e:
        self.logger.error(f"예상치 못한 오류: {e}")
        raise VideoProcessingError("비디오 처리 중 오류 발생") from e
```

### 나쁜 예시
```python
# 피해야 할 코드
try:
    # 위험한 작업들
except:
    pass  # 조용히 무시
```

## 테스트 작성

### 테스트 구조
```
tests/
├── test_elbow_angle_calculator.py
├── test_pitcher_identifier.py
├── test_pose_detector.py
├── test_release_detector.py
├── test_video_processor.py
└── integration/
    └── test_full_pipeline.py
```

### 테스트 작성 원칙
1. **단위 테스트**부터 작성 (개별 함수/메서드)
2. **통합 테스트**로 전체 플로우 검증
3. **엣지 케이스** 포함
4. **실패 시나리오** 테스트

### 테스트 예시
```python
class TestElbowAngleCalculator(unittest.TestCase):

    def setUp(self):
        """테스트 설정"""
        self.calculator = ElbowAngleCalculator()

    def test_calculate_angle_valid_input(self):
        """유효한 입력으로 각도 계산 테스트"""
        shoulder = {'x': 0, 'y': 0, 'z': 0}
        elbow = {'x': 1, 'y': 0, 'z': 0}
        wrist = {'x': 2, 'y': 0, 'z': 0}

        angle = self.calculator.calculate_elbow_angle(shoulder, elbow, wrist)

        self.assertIsNotNone(angle)
        self.assertAlmostEqual(angle, 180.0, places=1)

    def test_calculate_angle_invalid_input(self):
        """유효하지 않은 입력 테스트"""
        angle = self.calculator.calculate_elbow_angle(
            {'x': 0, 'y': 0, 'z': 0},
            {'x': 0, 'y': 0, 'z': 0},  # 동일한 점
            {'x': 0, 'y': 0, 'z': 0}
        )

        self.assertIsNone(angle)
```

### 테스트 실행
```bash
# 모든 테스트 실행
pytest tests/

# 특정 테스트 파일 실행
pytest tests/test_elbow_angle_calculator.py

# 커버리지 측정
pytest --cov=src tests/

# 상세한 출력
pytest -v tests/
```

## 커밋 메시지 규칙

### 커밋 메시지 형식
```
<type>(<scope>): <subject>

<body>

<footer>
```

### 타입 종류
- **feat**: 새로운 기능 추가
- **fix**: 버그 수정
- **docs**: 문서 업데이트
- **style**: 코드 포맷팅, 세미콜론 누락 등
- **refactor**: 코드 리팩토링 (기능 변경 없음)
- **test**: 테스트 코드 추가/수정
- **chore**: 빌드 작업, 의존성 업데이트 등

### 예시
```
feat(analysis): 팔꿈치 각도 계산 알고리즘 개선

- 2D 평면 기반 계산으로 정확도 향상
- 테스트 케이스 추가
- 성능 벤치마크 결과 포함

Closes #123
```

```
fix(pose-detector): MediaPipe 결과 파싱 오류 수정

- NormalizedLandmarkList 처리 로직 개선
- 예외 처리 강화
- 로그 메시지 개선
```

### 커밋 가이드라인
1. **제목은 50자 이내**로 작성
2. **본문은 72자 이내** 각 줄로 작성
3. **과거시제** 사용하지 말고 명령조로 작성
4. **첫 글자는 대문자**로 시작

## 코드 리뷰 체크리스트

### 기능 요구사항
- [ ] 요구사항이 올바르게 구현되었는가?
- [ ] 엣지 케이스가 처리되었는가?
- [ ] 오류 처리가 적절한가?

### 코드 품질
- [ ] 코드 스타일이 일관적인가?
- [ ] 변수/함수명이 의미 있는가?
- [ ] 중복 코드가 없는가?
- [ ] 하드코딩된 값이 없는가?

### 문서화
- [ ] 독스트링이 작성되었는가?
- [ ] 복잡한 로직에 주석이 있는가?
- [ ] API 문서가 업데이트되었는가?

### 테스트
- [ ] 단위 테스트가 작성되었는가?
- [ ] 통합 테스트가 있는가?
- [ ] 테스트 커버리지가 충분한가?

### 성능
- [ ] 불필요한 연산이 없는가?
- [ ] 메모리 누수가 없는가?
- [ ] 시간 복잡도가 적절한가?

## 개발 워크플로우

### 브랜치 전략
```
main (배포 브랜치)
├── develop (개발 브랜치)
    ├── feature/new-analysis-algorithm (기능 개발)
    ├── bugfix/pose-detection-error (버그 수정)
    └── hotfix/critical-performance-issue (긴급 수정)
```

### 작업 프로세스
1. **이슈 생성** → 기능 요구사항 정리
2. **브랜치 생성** → `feature/기능명` 형태로 생성
3. **개발** → 기능 구현 및 테스트 작성
4. **코드 리뷰** → 팀원 리뷰 요청
5. **병합** → develop 브랜치에 병합
6. **배포** → main 브랜치 배포

### 풀 리퀘스트 템플릿
```markdown
## 변경 사항
<!-- 무엇을 변경했는지 설명 -->

## 테스트 방법
<!-- 어떻게 테스트했는지 설명 -->

## 관련 이슈
<!-- 관련 이슈 번호 링크 -->

## 체크리스트
- [ ] 코드 리뷰 완료
- [ ] 테스트 통과
- [ ] 문서 업데이트
- [ ] 마이그레이션 스크립트 작성 (필요시)
```

이 가이드라인을 따라주시면 코드의 일관성과 품질을 유지할 수 있습니다. 추가 질문이 있으시면 언제든지 문의해주세요! 🚀
