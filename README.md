English | [한국어](README.ko.md)

# StoneSpring

> A project on the theme of *Artificial Intelligence*, developed as part of *(KDT) ROS2 and AI-based Autonomous Robot Developer Training Program, 8th Cohort*.


A caring chatbot GUI program that eases the user's loneliness, provides everyday information like a personal assistant, and looks after their mental health. \
It was developed with the future goal of combining with an actual care robot to provide emotional connection and practical help at the same time.
- An emotion-care chatbot based on user emotion analysis
- Facial/voice recognition, natural language processing, and schedule management, all integrated into one GUI
- A PyQt5-based desktop app + GPT-4o-mini streaming responses + DeepFace/Whisper-based emotion recognition


## Project Overview

- **Duration**: 2025.02.27 ~ 2025.04.07 (5 weeks / team project)
- **Team name**: 낭만 (NangMan; "Romance")


| Role   | Name (GitHub)  | Work |
|--------|--------|------|
| Team Lead   | Yeonwoo Gim ([@mumallaeng](https://github.com/mumallaeng)) | Initial project design<br/>Built the PyQt chat GUI<br /> Implemented real-time AI chat features (TTS, STT, streaming output, etc.) |
| Member   | Deokyun Na ([@YuSoYu](https://github.com/YuSoYu)) | Developed the voice emotion model (CNN+LSTM)<br/>MFCC extraction and data preprocessing<br/>Microphone testing and performance improvement |
| Member   | Chaehun Sim ([@Huni0128](https://github.com/Huni0128)) | Modularized gesture recognition<br/>Developed the MLP/SVC facial emotion model<br/>Integrated it into the PyQt-based GUI |
| Member   | Donguk Lim ([@Donguk-popo](https://github.com/Donguk-popo)) | Trained the gesture model |


### Tech Stack

| Category | Elements |
|------|-----------|
| **Language** | Python 3.12 |
| **Desktop UI** | PyQt5 |
| **Dialogue engine** | OpenAI Responses API (GPT-4o-mini), python-dotenv |
| **Voice interface** | whispercpp_kit (Whisper large-v2), PyAudio, gTTS, playsound |
| **Vision/Gesture** | OpenCV, Mediapipe, DeepFace(ArcFace) + scikit-learn MLP |
| **Voice emotion analysis** | TensorFlow/Keras, librosa |
| **Database** | MySQL, mysql-connector-python |
| **Infrastructure/Communication** | Python socket TCP/UDP server |



## Results and Materials

**Design Documents**
- [Software Requirement Specification (SRS)](https://github.com/addinedu-ros-8th/deeplearning-repo-3/wiki/%EC%86%8C%ED%94%84%ED%8A%B8%EC%9B%A8%EC%96%B4-%EC%9A%94%EA%B5%AC%EC%82%AC%ED%95%AD-%EB%AA%85%EC%84%B8%EC%84%9C-(SRS))
- [Interface Specification](https://github.com/addinedu-ros-8th/deeplearning-repo-3/wiki/Interface-Specification)
- [GUI Specification](https://github.com/addinedu-ros-8th/deeplearning-repo-3/wiki/GUI-Structure)


### Implementation


![Full Operation](https://github.com/user-attachments/assets/6c8050f3-1440-40ec-9f67-f2b3d8110960)








![Full AI](https://github.com/user-attachments/assets/0dcbac9b-3d1b-4d94-b31f-684573bd12fb)

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/8de988a7-1bd0-4fe3-8822-911b8467bb7f" alt="Mic + Chat 1" /></td>
    <td><img src="https://github.com/user-attachments/assets/e94d3320-9f44-4385-926b-8689eeeaa983" alt="Expression" /></td>
  </tr>    

  <tr>
    <td><img src="https://github.com/user-attachments/assets/34d1ac60-6068-4ca0-820f-f45152096092" alt="Gesture 1" /></td>
    <td><img src="https://github.com/user-attachments/assets/4cd97090-64e2-468c-92ba-cdb817604f54" alt="Gesture 2" /></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/b354fca6-9685-48c3-8e9c-8732424fa46f" alt="Gesture 3" /></td>
    <td><img src="https://github.com/user-attachments/assets/5f9de301-077c-4451-9c32-01432257c53c" alt="Gesture 4" /></td>
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

- Dataset: AI Hub `Emotional Speech Synthesis Data by Speaking Style` (7 emotion labels)
- Preprocessing:
    - Converted JSON into a Pandas DataFrame to normalize metadata and file paths
    - After correcting class imbalance, extracted MFCC (40×100) spectra and cached them as `.npy`
    - Integer-encoded emotion labels and mapped them to the Softmax output
- Model training:
    - A TensorFlow-based CNN with 3 blocks of Conv2D → BatchNorm → MaxPool plus Dropout
    - A classifier taking input shape `(40, 100, 1)`, with the final stage being Dense + Softmax(len(labels))
    - Batch loading via `MFCCGenerator`, with EarlyStopping(patience=30) and ModelCheckpoint for training stability
- Evaluation:
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
    - Monitored the train/validation loss curves and adopted the model with the lowest `val_loss`
    - The 7-class training converges stably around epoch 100–300, and per-class misclassification was reviewed via the confusion matrix

<br/>

**Face Emotion Model**

![face_emotion](https://github.com/user-attachments/assets/61e4c2d7-6803-4332-81af-7d1b1a67aa99)

- Dataset: AI Hub `Korean Facial Expression Data for Emotion Recognition` (4 classes: happy, sad, angry, neutral)
- Preprocessing:
    - Downsampled to 5,000 images per class to balance the dataset
    - Detected face regions with MTCNN, then removed tilted samples using EYE_THRESHOLD
    - Resized to a maximum width of 640px and saved the cropped images
- Model training:
    - Generated 512-dimensional face embeddings with DeepFace ArcFace
    - Classified emotions using scikit-learn's `MLPClassifier(hidden_layer_sizes=(128, 64))`
    - Stabilized the training pipeline with batch processing and intermediate checkpoints
- Evaluation:
    - Produced a `classification_report` on an 8:2 train/validation split
  <table>
    <tr>
      <td><img width="640" height="480" alt="face_emotion_model-1" src="https://github.com/user-attachments/assets/7a371d42-ef53-49e9-bbd7-05a50331c946" /></td>
      <td><img width="600" height="500" alt="face_emotion_model-2" src="https://github.com/user-attachments/assets/dba18366-505b-4414-97ae-8f022595fd3c" /></td>
    </tr>
  </table>
    - The happy class scored the best with F1 ≈ 0.89; angry/sad ranged 0.75–0.78 with mutual misclassification between them
    - Analyzed the confusion matrix and planned to augment the angry/sad data

<br/>


**Gesture Recognition Model**

![gesture](https://github.com/user-attachments/assets/afe35e53-65ef-44eb-bfb2-08a05c1ef85b)

- Data collection:
    - When a user records a gesture for 10 seconds, MediaPipe Hands extracts 21 hand landmarks
    - Normalized hand coordinates and joint angles into a 99-feature vector for a 30-frame sequence
    - New gestures can be added and the model fully retrained based on the accumulated `.npy` data
- Model training:
    - LSTM(64, ReLU) → Dense(32, ReLU) → Softmax(len(actions)) architecture taking input `(30, 99)`
    - Saved the trained weights for real-time inference
- Evaluation:
    <img width="1189" height="590" alt="gesture-model" src="https://github.com/user-attachments/assets/6dd93f00-cfbe-4a39-a9f3-91bfbe155f34" />
    - In real-time testing with `gesture_recognize.py`, results with confidence below 0.8 are ignored, and a gesture is only confirmed once the same inference holds for 3 consecutive frames
    - Verified per-user recognition accuracy through repeated demonstrations under varying lighting and background noise conditions

<br/><br/><br/>


### System Architecture

**Final Goal**

<img width="1342" height="1304" alt="system_architecture" src="https://github.com/user-attachments/assets/9338e507-90f2-4a2b-bcde-e46244e95df5" />

<table>
  <tr>
    <td><div>v0.1.0</div><img width="964" height="625" alt="system_architecture_0 1 0" src="https://github.com/user-attachments/assets/9c4f1c9f-fe3b-4aea-8269-1d31b73fe3a7" /></td>
    <td><div>v0.2.0</div><img width="1054" height="444" alt="system_architecture_0 2 0" src="https://github.com/user-attachments/assets/c6a0802a-d7f6-4f65-9540-92d96f49bba5" /></td>
    <td><div>v0.3.0</div><img width="945" height="765" alt="system_architecture_0 3 0" src="https://github.com/user-attachments/assets/ab7c4907-e848-4ec9-917a-27735a0b1799" /></td>
  </tr>
</table>


### Data Structure

[Reference material](https://github.com/addinedu-ros-8th/deeplearning-repo-3/wiki/%EB%8D%B0%EC%9D%B4%ED%84%B0-%EA%B5%AC%EC%A1%B0-%EC%B0%B8%EA%B3%A0-%EC%9E%90%EB%A3%8C)

<img width="2160" height="2082" alt="data_structure" src="https://github.com/user-attachments/assets/fa27f37a-c672-40be-b6ed-12fc9f636d69" />

<table>
  <tr>
    <td><img width="1278" height="1312" alt="data_structure_1" src="https://github.com/user-attachments/assets/53cf7340-fd70-4858-9208-d503022752e5" /></td>
    <td><img width="1268" height="1694" alt="data_structure_2" src="https://github.com/user-attachments/assets/a9a5daec-c050-4fb1-a389-adee8604d8c7" /></td>
    <td><img width="1200" height="640" alt="data_structure_3" src="https://github.com/user-attachments/assets/ebd618ef-2eb0-4d11-84a4-d597554bd1b2" /></td>
  </tr>
</table>



## Installation and Usage

```shell
bash run.sh
```
