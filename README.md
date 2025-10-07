# 야구 투구 분석 시스템 (Baseball Pitch Analysis System)

이 프로젝트는 야구 투구 비디오를 분석하여 각 투구 시점의 팔꿈치 각도를 추출하는 컴퓨터 비전 시스템입니다. MediaPipe 포즈 감지 기술을 사용하여 투수의 동작을 분석하고, 공 릴리스 시점의 팔꿈치 각도를 측정합니다.

## 🚀 주요 기능

- **다중 포즈 감지**: MediaPipe를 사용하여 비디오의 각 프레임에서 여러 사람의 포즈를 감지
- **투수 식별**: 화면에서 투수만을 정확하게 식별하고 추적
- **팔꿈치 각도 계산**: 3차원 공간에서 어깨-팔꿈치-손목 각도를 계산
- **공 릴리스 시점 감지**: 투구 동작에서 공을 놓는 순간을 자동으로 식별
- **비디오 주석**: 분석 결과가 시각적으로 표시된 주석 비디오 생성
- **CSV 결과 출력**: 분석 데이터를 구조화된 CSV 파일로 저장

## 📋 요구사항

### 필수 라이브러리
- Python 3.7+
- mediapipe
- opencv-python
- numpy
- pandas

### 설치 방법
```bash
pip install mediapipe opencv-python numpy pandas
```

## 🎯 사용법

### 전체 분석 실행
```bash
python main.py <입력_디렉토리> [출력_디렉토리]
```

예시:
```bash
python main.py webb_logan_09_28
python main.py webb_logan_09_28 output_results
```

### 단일 비디오 테스트
```bash
python main.py test <비디오_파일> [출력_파일]
```

예시:
```bash
python main.py test 001-FC-90.9.mp4
python main.py test 001-FC-90.9.mp4 test_output.mp4
```

## 📁 입력 파일 형식

프로젝트는 다음 형식의 MP4 비디오 파일들을 처리합니다:

- 파일명 형식: `{투구번호}-{구종}-{구속}.mp4`
- 예시: `001-FC-90.9.mp4`, `002-SI-93.4.mp4`

### 구종 코드
- `FF`: Fastball (패스트볼)
- `SI`: Sinker (싱커)
- `FC`: Cutter (커터)
- `CH`: Changeup (체인지업)
- `SL`: Slider (슬라이더)
- `CB`: Curveball (커브볼)
- `ST`: Splitter (스플리터)

## 📊 출력 결과

### CSV 파일 (`pitch_analysis_results.csv`)
| 컬럼 | 설명 |
|------|------|
| `pitch_number` | 투구 번호 |
| `pitch_type` | 구종 |
| `velocity_mph` | 구속 (mph) |
| `elbow_angle_degrees` | 팔꿈치 각도 (도) |
| `throwing_arm` | 투구 팔 (LEFT/RIGHT) |
| `release_frame` | 공 릴리스 프레임 번호 |
| `motion_quality` | 투구 동작 품질 |

### 주석 비디오
- 원본 파일명에 `_processed` 접미사가 추가됩니다
- 예시: `001-FC-90.9_processed.mp4`
- 비디오에는 다음 정보가 실시간으로 표시됩니다:
  - 포즈 랜드마크와 연결선
  - 현재 팔꿈치 각도
  - 투구 팔 정보
  - 프레임 번호

## 🏗️ 프로젝트 구조

```
pitcheezy_for_mlb/
├── main.py                    # 메인 실행 스크립트
├── pose_detector.py           # MediaPipe 포즈 감지 모듈
├── pitcher_identifier.py      # 투수 식별 모듈
├── elbow_angle_calculator.py  # 팔꿈치 각도 계산 모듈
├── release_detector.py        # 공 릴리스 시점 감지 모듈
├── video_processor.py         # 비디오 처리 및 주석 모듈
├── README.md                  # 프로젝트 설명서
└── webb_logan_09_28/         # 입력 비디오 파일들
    ├── 001-FC-90.9.mp4
    ├── 002-SI-93.4.mp4
    └── ...
```

## 🔧 기술 세부사항

### 투수 식별 알고리즘
1. **화면 중앙 영역 필터링**: 화면 중앙 30% 영역에 있는 사람들을 우선적으로 고려
2. **투구 동작 감지**: 팔 움직임 패턴을 분석하여 투구 동작 식별
3. **최적 투수 선택**: 가장 강한 투구 동작을 보이는 사람을 투수로 선택

### 팔꿈치 각도 계산
- **3차원 벡터 계산**: 어깨, 팔꿈치, 손목 세 점의 3D 좌표 사용
- **벡터 정규화**: 코사인 각도 계산을 위한 벡터 정규화
- **수치적 안정성**: 각도 계산 시 예외 처리 및 클리핑 적용

### 공 릴리스 시점 감지
- **팔 펴짐 분석**: 팔꿈치-손목 거리와 어깨-팔꿈치 거리의 비율 분석
- **각도 최대점 찾기**: 팔꿈치 각도가 최대인 지점을 릴리스 시점으로 판단
- **동작 품질 평가**: 각도 변화의 부드러움으로 투구 품질 평가

## ⚙️ 설정 파라미터

각 모듈에서 조정 가능한 주요 파라미터들:

### PoseDetector
- `min_detection_confidence`: 최소 감지 신뢰도 (기본값: 0.5)
- `min_tracking_confidence`: 최소 추적 신뢰도 (기본값: 0.5)

### PitcherIdentifier
- `center_region_ratio`: 화면 중앙 영역 비율 (기본값: 0.3)
- `min_arm_movement_threshold`: 최소 팔 움직임 임계값 (기본값: 50)

### ReleaseDetector
- `arm_extension_threshold`: 팔 펴짐 임계값 (기본값: 0.9)
- `min_angle_threshold`: 최소 팔꿈치 각도 (기본값: 160)
- `stability_window`: 안정성 확인 윈도우 크기 (기본값: 3)

## 🚨 주의사항

- 비디오 파일은 MP4 형식이어야 합니다
- 분석 정확도는 비디오 화질과 조명 상태에 영향을 받습니다
- 복잡한 장면(여러 선수가 동시에 움직이는 경우)에서는 정확도가 떨어질 수 있습니다
- 대용량 비디오 파일의 경우 처리 시간이 오래 걸릴 수 있습니다

## 🔍 문제 해결

### 일반적인 문제들
1. **포즈가 감지되지 않음**: `min_detection_confidence` 값을 낮춰보세요 (0.3-0.5)
2. **투수가 잘못 식별됨**: `center_region_ratio` 값을 조정하거나 투구 동작 임계값을 확인하세요
3. **릴리스 시점이 부정확함**: `arm_extension_threshold`와 각도 임계값을 조정하세요

### 로그 확인
프로그램 실행 시 자세한 로그가 출력됩니다. 오류 발생 시 로그를 확인하여 문제의 원인을 파악하세요.

## 📈 향후 개선사항

- [ ] GPU 가속 지원
- [ ] 실시간 분석 기능
- [ ] 더 정교한 투수 식별 알고리즘
- [ ] 추가적인 생체역학 지표 계산
- [ ] 웹 인터페이스 개발

## 📄 라이선스

이 프로젝트는 연구 및 교육 목적으로 개발되었습니다. 사용 시 출처를 밝혀주세요.

## 👥 개발자

이 프로젝트는 야구 데이터 분석을 위한 컴퓨터 비전 시스템으로 개발되었습니다.

---

**시작하기**: `python main.py webb_logan_09_28` 명령어로 분석을 시작해보세요!
