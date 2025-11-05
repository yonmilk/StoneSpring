# StoneSpring

> *(KDT) ROS2와 인공지능을 활용한 자율주행 로봇 개발자 양성과정 8기* 교육과정 내에서 *Artificial Intelligence*을 주제로한 과제입니다.


사용자의 외로움을 덜어주고, 일상생활에서 필요한 정보를 비서처럼 알려주며, 정신 건강을 챙겨주는 돌봄 챗봇 GUI 프로그램입니다. \
향후에는 실제 돌봄 로봇과 결합하여 정서적 교감과 실질적 도움을 동시에 제공하는 통합 서비스를 목표로 개발했습니다.
- 사용자 감정 분석 기반 정서 케어 챗봇
- 표정/음성 인식, 자연어 처리, 일정 관리까지 GUI 통합 제공
- PyQt5 기반 데스크탑 앱 + GPT-4o-mini 스트리밍 응답 + DeepFace/Whisper 기반 감정 인식


## 프로젝트 개요
- **기간**: 2025.02.27 ~ 2025.04.07 (5주)
- **팀명**: 낭만(浪漫; NangMan)


| 역할   | 이름 (Github)  | 작업 |
|--------|--------|------|
| 팀장   | 김연우  ([@yonmilk](https://github.com/yonmilk)) | 프로젝트 초기 설계<br/>PyQt 채팅 GUI 제작<br /> 실시간 AI 채팅 기능(TTS, STT, 스트리밍 출력 등) 구현 |
| 팀원   | 나덕윤 ([@YuSoYu](https://github.com/YuSoYu)) | 음성 감정 모델 개발 (CNN+LSTM)<br/>MFCC 추출 및 데이터 전처리<br/>마이크 테스트 및 성능 개선 |
| 팀원   | 심채훈 ([@Huni0128](https://github.com/Huni0128)) | 제스처 인식 모듈화<br/>MLP/SVC 얼굴 감정 모델 개발<br/>PyQt 기반 GUI 통합 |
| 팀원   | 임동욱 ([@Donguk-popo](https://github.com/Donguk-popo)) | 제스처 모델 학습 |


## 기술 스택
| 분류 | 기술 요소 |
|------|-----------|
| **언어** | Python 3.12 |
| **데스크톱 UI** | PyQt5, Qt Widgets/QSS |
| **대화 엔진** | OpenAI Responses API (GPT-4o-mini), python-dotenv |
| **음성 인터페이스** | whispercpp_kit (Whisper large-v2), PyAudio, gTTS, playsound |
| **비전/제스처** | OpenCV, Mediapipe, DeepFace(ArcFace) + scikit-learn MLP |
| **음성 감정 분석** | TensorFlow/Keras, librosa |
| **데이터베이스** | MySQL, mysql-connector-python |
| **인프라/통신** | Python socket TCP/UDP 서버, Docker (python:3.12-slim) |


### AI 채팅 시스템 개발
- OpenAI API를 사용한 GPT-4o-mini 모델 스트리밍 출력 구현
- 마크다운 지원 응답 처리 로직 개발
- 케어 상황에 맞는 답변 시나리오 강화 및 학습 데이터 정제

### 실시간 통신 시스템 구축
- TCP/UDP 양방향 통신 기반 실시간 챗봇 메시지 송수신 구성
- 채팅 UI/UX 개선을 위한 통신 최적화

### 음성 인터페이스 개발
- Whisper.cpp 기반 STT 모델(Large-v2) 연동
- 실시간 음성 인식 처리 시스템 구현
- Google TTS(gTTS)를 활용한 감성 대화용 음성 안내 기능 구성

### 프로젝트 관리 및 인프라
- 초기 프로젝트 구조 설계 및 모듈 분담
- 실행 자동화 스크립트(run.sh) 구성으로 팀원 간 로컬 테스트 환경 통일
- 프로젝트 전반에 대한 리팩토링 및 릴리즈 관리

## 핵심 성과
- 자연스러운 대화 흐름을 지원하는 실시간 채팅 시스템 구현
- 음성 입출력을 통한 직관적인 사용자 인터페이스 제공
- 돌봄 상황에 특화된 맞춤형 AI 응답 시스템 개발
- 안정적인 실시간 통신 및 스트리밍 처리 시스템 구축




## 프로젝트 결과 및 자료

### 구현

![전체동작](https://github.com/user-attachments/assets/6c8050f3-1440-40ec-9f67-f2b3d8110960)


![AI전체](https://github.com/user-attachments/assets/0dcbac9b-3d1b-4d94-b31f-684573bd12fb)


**Voice Emotion Model**

![마이크+채팅1](https://github.com/user-attachments/assets/8de988a7-1bd0-4fe3-8822-911b8467bb7f)




**Face Emotion Model and Gesture Recognition Model**

![표정](https://github.com/user-attachments/assets/e94d3320-9f44-4385-926b-8689eeeaa983)

![제스처1](https://github.com/user-attachments/assets/34d1ac60-6068-4ca0-820f-f45152096092)

![제스처2](https://github.com/user-attachments/assets/4cd97090-64e2-468c-92ba-cdb817604f54)

![제스처3](https://github.com/user-attachments/assets/b354fca6-9685-48c3-8e9c-8732424fa46f)

![제스처4](https://github.com/user-attachments/assets/5f9de301-077c-4451-9c32-01432257c53c)


#### 아쉬웠던 점 및 개선 방향



### System Architecture

최종 목표

<ImgTag src="https://drive.google.com/file/d/1yAQRxzdL5doOcbeGXAQM2NyC99OfPBaI/view?usp=sharing" alt="StoneSpring System Architecture" />



아래부터는 구현된 버전에 대한 시스템 아키텍처입니다.

#### ver 0.1.0




#### ver 0.2.0

#### ver 0.3.0


### Data Structure


### Sequence Diagram



## 설치 및 실행 방법

```shell
bash run.sh
```
