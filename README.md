# StoneSpring

> *(KDT) ROS2와 인공지능을 활용한 자율주행 로봇 개발자 양성과정 8기* 교육과정 내에서 *Artificial Intelligence*을 주제로한 과제입니다.


사용자의 외로움을 덜어주고, 일상생활에서 필요한 정보를 비서처럼 알려주며, 정신 건강을 챙겨주는 돌봄 챗봇 GUI 프로그램입니다. \
향후에는 실제 돌봄 로봇과 결합하여 정서적 교감과 실질적 도움을 동시에 제공하는 통합 서비스를 목표로 개발했습니다.
- 사용자 감정 분석 기반 정서 케어 챗봇
- 표정/음성 인식, 자연어 처리, 일정 관리까지 GUI 통합 제공
- PyQt5 기반 데스크탑 앱 + GPT-4o-mini 스트리밍 응답 + DeepFace/Whisper 기반 감정 인식


## 프로젝트 개요

- **기간**: 2025.02.27 ~ 2025.04.07 (5주/팀프로젝트)
- **팀명**: 낭만(浪漫; NangMan)


| 역할   | 이름 (Github)  | 작업 |
|--------|--------|------|
| 팀장   | 김연우  ([@yonmilk](https://github.com/yonmilk)) | 프로젝트 초기 설계<br/>PyQt 채팅 GUI 제작<br /> 실시간 AI 채팅 기능(TTS, STT, 스트리밍 출력 등) 구현 |
| 팀원   | 나덕윤 ([@YuSoYu](https://github.com/YuSoYu)) | 음성 감정 모델 개발 (CNN+LSTM)<br/>MFCC 추출 및 데이터 전처리<br/>마이크 테스트 및 성능 개선 |
| 팀원   | 심채훈 ([@Huni0128](https://github.com/Huni0128)) | 제스처 인식 모듈화<br/>MLP/SVC 얼굴 감정 모델 개발<br/>PyQt 기반 GUI 통합 |
| 팀원   | 임동욱 ([@Donguk-popo](https://github.com/Donguk-popo)) | 제스처 모델 학습 |


### 기술 스택

| 분류 | 기술 요소 |
|------|-----------|
| **언어** | Python 3.12 |
| **데스크톱 UI** | PyQt5 |
| **대화 엔진** | OpenAI Responses API (GPT-4o-mini), python-dotenv |
| **음성 인터페이스** | whispercpp_kit (Whisper large-v2), PyAudio, gTTS, playsound |
| **비전/제스처** | OpenCV, Mediapipe, DeepFace(ArcFace) + scikit-learn MLP |
| **음성 감정 분석** | TensorFlow/Keras, librosa |
| **데이터베이스** | MySQL, mysql-connector-python |
| **인프라/통신** | Python socket TCP/UDP 서버 |



## 프로젝트 결과 및 자료

**설계 자료**
- [Software Requirement Specification (SRS)](https://github.com/addinedu-ros-8th/deeplearning-repo-3/wiki/%EC%86%8C%ED%94%84%ED%8A%B8%EC%9B%A8%EC%96%B4-%EC%9A%94%EA%B5%AC%EC%82%AC%ED%95%AD-%EB%AA%85%EC%84%B8%EC%84%9C-(SRS))
- [Interface Specification](https://github.com/addinedu-ros-8th/deeplearning-repo-3/wiki/Interface-Specification)
- [GUI Specification](https://github.com/addinedu-ros-8th/deeplearning-repo-3/wiki/GUI-Structure)


### 구현


![전체동작](https://github.com/user-attachments/assets/6c8050f3-1440-40ec-9f67-f2b3d8110960)








![AI전체](https://github.com/user-attachments/assets/0dcbac9b-3d1b-4d94-b31f-684573bd12fb)

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/8de988a7-1bd0-4fe3-8822-911b8467bb7f" alt="마이크+채팅1" /></td>
    <td><img src="https://github.com/user-attachments/assets/e94d3320-9f44-4385-926b-8689eeeaa983" alt="표정" /></td>
  </tr>    

  <tr>
    <td><img src="https://github.com/user-attachments/assets/34d1ac60-6068-4ca0-820f-f45152096092" alt="제스처1" /></td>
    <td><img src="https://github.com/user-attachments/assets/4cd97090-64e2-468c-92ba-cdb817604f54" alt="제스처2" /></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/b354fca6-9685-48c3-8e9c-8732424fa46f" alt="제스처3" /></td>
    <td><img src="https://github.com/user-attachments/assets/5f9de301-077c-4451-9c32-01432257c53c" alt="제스처4" /></td>
  </tr>
</table>

<table>
  <tr>
    <td><img width="812" height="944" alt="GUI-1" src="https://github.com/user-attachments/assets/727c8bba-e7ee-4d9d-acdf-89890ce06d35" /></td>
    <td><img width="462" height="429" alt="GUI-2" src="https://github.com/user-attachments/assets/cfd2effd-db09-4302-ac78-0af839638d0e" /></td>
    <td><img width="812" height="944" alt="GUI-3" src="https://github.com/user-attachments/assets/9a204c07-a5ae-4332-9a0f-6a71ad2f22a1" /></td>
  </tr>
  <tr>
    <td><img width="812" height="944" alt="GUI-4" src="https://github.com/user-attachments/assets/37fe0616-0042-4dec-8a69-2a0bf05cce93" /></td>
    <td><img width="612" height="544" alt="GUI-5" src="https://github.com/user-attachments/assets/132d82f7-a984-4e0c-b4e3-af98bfa880ba" /></td>
    <td><img width="812" height="944" alt="GUI-6" src="https://github.com/user-attachments/assets/5bf33799-a90d-44b7-bffb-c761b25546a1" /></td>
  </tr>
  <tr>
    <td><img width="812" height="944" alt="GUI-7" src="https://github.com/user-attachments/assets/42de39c4-9a43-4496-8320-6eef5eaeb784" /></td>
    <td><img width="812" height="944" alt="GUI-8" src="https://github.com/user-attachments/assets/9892507e-a484-4dd3-9d64-819ef9e6711b" /></td>
    <td></td>
  </tr>
</table>


<br/>

**Voice Emotion Model**

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/a27f86b5-0abe-4e4a-b2e4-ba876b7c586d" alt="voice_emotion-1" /></td>
    <td><img src="https://github.com/user-attachments/assets/16f9c737-db04-433e-8978-d82c488777d6" alt="voice_emotion-2" /></td>
    <td></td>
  </tr>
</table>

- 데이터셋: AI Hub `감성 및 발화 스타일별 음성합성 데이터` (7개 감정 레이블 구성)
- 전처리:
    - JSON을 Pandas DataFrame으로 변환해 메타데이터와 파일 경로를 정규화
    - 클래스 불균형을 조정한 뒤 MFCC(40×100) 스펙트럼을 추출하고 `.npy`에 캐시
    - 감정 라벨을 정수 인코딩하여 Softmax 출력과 매핑
- 모델 학습:
    - TensorFlow 기반 Conv2D → BatchNorm → MaxPool 블록 3단과 Dropout을 포함한 CNN 구조
    - 입력 형태 `(40, 100, 1)`을 사용하는 분류기, 마지막 단계는 Dense + Softmax(len(labels))
    - `MFCCGenerator`로 배치 단위 로딩, EarlyStopping(patience=30)과 ModelCheckpoint로 학습 안정화
- 평가:
  <table>
    <tr>
      <td><img width="1979" height="780" alt="voice_emotion_model-1" src="https://github.com/user-attachments/assets/806f7d21-405a-48b8-9817-c2ebe6ec5c2a" /></td>
      <td><img width="1389" height="490" alt="voice_emotion_model-2" src="https://github.com/user-attachments/assets/13e8ea25-c8cc-47c7-9ef2-70353f106061" /></td>
    </tr>
    <tr>
      <td><img width="1389" height="490" alt="voice_emotion_model-3" src="https://github.com/user-attachments/assets/6e905ef4-f75d-4acc-94be-c83d156a4b4a" /></td>
      <td><img width="1189" height="490" alt="voice_emotion_model-4" src="https://github.com/user-attachments/assets/f6a55d25-2644-4183-80b7-50b08bfce0ac" /></td>
    </tr>
  </table>
    - 학습/검증 손실 곡선을 모니터링하며 최저 `val_loss` 모델을 채택
    - 7-class 학습에서 epoch 100~300 구간에서 안정 수렴하며, confusion matrix로 클래스별 오분류를 검토

<br/>

**Face Emotion Model**

![face_emotion](https://github.com/user-attachments/assets/61e4c2d7-6803-4332-81af-7d1b1a67aa99)

- 데이터셋: AI Hub `한국인 감정인식을 위한 복합 영상` (행복·슬픔·분노·중립 4클래스)
- 전처리:
    - 클래스별 5,000장으로 다운샘플링하여 데이터 균형 확보
    - MTCNN으로 얼굴 영역을 검출한 뒤 EYE_THRESHOLD로 기울어진 샘플 제거
    - 최대 너비 640px 기준으로 리사이징하고, crop 이미지를 저장
- 모델 학습:
    - DeepFace ArcFace로 512차원 얼굴 임베딩을 생성
    - scikit-learn `MLPClassifier(hidden_layer_sizes=(128, 64))`를 사용해 감정 분류
    - 일괄 처리와 중간 저장으로 학습 파이프라인을 안정화
- 평가:
    - 8:2로 분할한 검증 세트에서 `classification_report`를 산출
  <table>
    <tr>
      <td><img width="640" height="480" alt="face_emotion_model-1" src="https://github.com/user-attachments/assets/7a371d42-ef53-49e9-bbd7-05a50331c946" /></td>
      <td><img width="600" height="500" alt="face_emotion_model-2" src="https://github.com/user-attachments/assets/dba18366-505b-4414-97ae-8f022595fd3c" /></td>
    </tr>
  </table>
    - happy 클래스 F1 ≈ 0.89로 최고 성능, angry/sad는 0.75~0.78 범위에서 상호 오분류가 발생
    - 혼동 행렬을 분석해 분노/슬픔 데이터 보강 계획을 수립

<br/>


**Gesture Recognition Model**

![gesture](https://github.com/user-attachments/assets/afe35e53-65ef-44eb-bfb2-08a05c1ef85b)

- 데이터 수집:
    - 사용자가 제스처를 10초간 촬영하면 MediaPipe Hands로 21개 손 관절을 추출
    - 손 좌표와 관절 각도를 정규화해 30프레임 시퀀스의 99개 특징 벡터로 변환
    - 누적 `.npy` 데이터를 기반으로 신규 제스처 추가 시 전체 재학습 가능
- 모델 학습:
    - 입력 `(30, 99)`을 사용하는 LSTM(64, ReLU) → Dense(32, ReLU) → Softmax(len(actions)) 구조
    - 학습 완료 후 실시간 추론에 사용할 가중치를 저장
- 평가:
    <img width="1189" height="590" alt="gesture-model" src="https://github.com/user-attachments/assets/6dd93f00-cfbe-4a39-a9f3-91bfbe155f34" />
    - `gesture_recognize.py` 실시간 테스트에서 확률 0.8 미만 결과를 무시하고, 동일 추론이 3프레임 연속일 때만 제스처로 확정
    - 조명 변화와 배경 노이즈 환경에서 반복 시연하며 사용자별 인식률을 검증

<br/><br/><br/>


### System Architecture

**최종 목표**

<img width="1342" height="1304" alt="system_architecture" src="https://github.com/user-attachments/assets/9338e507-90f2-4a2b-bcde-e46244e95df5" />

<table>
  <tr>
    <td><div>v0.1.0</div><img width="964" height="625" alt="system_architecture_0 1 0" src="https://github.com/user-attachments/assets/9c4f1c9f-fe3b-4aea-8269-1d31b73fe3a7" /></td>
    <td><div>v0.2.0</div><img width="1054" height="444" alt="system_architecture_0 2 0" src="https://github.com/user-attachments/assets/c6a0802a-d7f6-4f65-9540-92d96f49bba5" /></td>
    <td><div>v0.3.0</div><img width="945" height="765" alt="system_architecture_0 3 0" src="https://github.com/user-attachments/assets/ab7c4907-e848-4ec9-917a-27735a0b1799" /></td>
  </tr>
</table>


### Data Structure

[참고 자료](https://github.com/addinedu-ros-8th/deeplearning-repo-3/wiki/%EB%8D%B0%EC%9D%B4%ED%84%B0-%EA%B5%AC%EC%A1%B0-%EC%B0%B8%EA%B3%A0-%EC%9E%90%EB%A3%8C)

<img width="2160" height="2082" alt="data_structure" src="https://github.com/user-attachments/assets/fa27f37a-c672-40be-b6ed-12fc9f636d69" />

<table>
  <tr>
    <td><img width="1278" height="1312" alt="data_structure_1" src="https://github.com/user-attachments/assets/53cf7340-fd70-4858-9208-d503022752e5" /></td>
    <td><img width="1268" height="1694" alt="data_structure_2" src="https://github.com/user-attachments/assets/a9a5daec-c050-4fb1-a389-adee8604d8c7" /></td>
    <td><img width="1200" height="640" alt="data_structure_3" src="https://github.com/user-attachments/assets/ebd618ef-2eb0-4d11-84a4-d597554bd1b2" /></td>
  </tr>
</table>



### Sequence Diagram

<img width="1616" height="893" alt="sequence_diagram_1" src="https://github.com/user-attachments/assets/fb54ee8e-b56b-4a6e-af74-4c4cd09cafb1" />

<table>
  <tr>
    <td><img width="1163" height="572" alt="sequence_diagram_2" src="https://github.com/user-attachments/assets/e34831d3-ff66-4c7a-925e-013b16baa10e" /></td>
    <td><img width="1690" height="607" alt="sequence_diagram_3" src="https://github.com/user-attachments/assets/325dbfaa-1b34-4865-81c1-dfb173f4ce0a" /></td>
  </tr>
  <tr>
    <td><img width="1214" height="895" alt="sequence_diagram_4" src="https://github.com/user-attachments/assets/a98c8385-c77b-4f21-aa72-f8211441c04f" /></td>
    <td><img width="1837" height="643" alt="sequence_diagram_5" src="https://github.com/user-attachments/assets/6bbe332a-72aa-4fac-94f3-b39fdd9b4357" /></td>
  </tr>
</table>


## 설치 및 실행 방법

```shell
bash run.sh
```



