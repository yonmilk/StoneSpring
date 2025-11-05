import cv2
import mediapipe as mp
import numpy as np
import time, os
import json
from tensorflow.keras.utils import to_categorical   
from sklearn.model_selection import train_test_split    
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense         
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

def collect_and_train(gesture_name, data_path, model_path, secs_for_action=15):
    """
    사용자로부터 제스처 데이터를 수집하고 기존 제스처 파일들과 함께 모델을 학습합니다.
    """
    labels_path = os.path.join(data_path, "gesture_labels.json")
    actions = []
    if os.path.exists(labels_path):
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                existing_actions = json.load(f)
                if isinstance(existing_actions, list):
                    actions.extend(existing_actions)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[Gesture Train] 기존 레이블을 불러오는 데 실패했습니다: {e}")

    seq_length = 30    

    # MediaPipe hands model
    mp_hands = mp.solutions.hands      
    mp_drawing = mp.solutions.drawing_utils       
    hands = mp_hands.Hands(                     
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)
    created_time = int(time.time())             
    os.makedirs(data_path, exist_ok=True)         

    # 새로운 데이터 수집
    while cap.isOpened():
        for idx, action in enumerate([gesture_name]): 
            data = []
            ret, img = cap.read()                
            img = cv2.flip(img, 1)          
            cv2.putText(img, f'Collecting {action.upper()} action...', org=(10, 30), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, 
                        color=(255, 255, 255), thickness=2)
            cv2.imshow('img', img)       
            cv2.waitKey(3000)

            start_time = time.time()      
            while time.time() - start_time < secs_for_action:   
                ret, img = cap.read()                               
                img = cv2.flip(img, 1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        
                result = hands.process(img)                       
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                if result.multi_hand_landmarks is not None:      
                    for res in result.multi_hand_landmarks:       
                        joint = np.zeros((21, 4))               
                        for j, lm in enumerate(res.landmark):
                            joint[j] = [lm.x, lm.y, lm.z, lm.visibility]    
                        v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3]      
                        v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3]    
                        v = v2 - v1      
                        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]   
                        angle = np.arccos(np.einsum('nt,nt->n',
                            v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
                            v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]))         
                        angle = np.degrees(angle)         
                        angle_label = np.array([angle], dtype=np.float32)           
                        angle_label = np.append(angle_label, idx)                  
                        d = np.concatenate([joint.flatten(), angle_label])          
                        data.append(d)                                              
                        mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)      
                cv2.imshow('img', img)              
                if cv2.waitKey(1) == ord('q'):
                    return

            data = np.array(data)
            print(action, data.shape)
            np.save(os.path.join(data_path, f'raw_{action}_{created_time}'), data)
            full_seq_data = []
            for seq in range(len(data) - seq_length):
                full_seq_data.append(data[seq:seq + seq_length])
            full_seq_data = np.array(full_seq_data)
            print(action, full_seq_data.shape)
            np.save(os.path.join(data_path, f'seq_{action}_{created_time}'), full_seq_data)
        break
    cap.release()
    cv2.destroyAllWindows()

    # 기존 데이터 로드 및 병합
    x_data = []
    y_data = []
    for filename in sorted(os.listdir(data_path)):
        if filename.startswith('seq_') and filename.endswith('.npy'):
            action_name = filename.split('_')[1]
            if action_name not in actions:
                actions.append(action_name)
            action_index = actions.index(action_name)
            data = np.load(os.path.join(data_path, filename))
            x_data.extend(data[:, :, :-1].tolist())
            y_data.extend([action_index] * len(data))
    x_data = np.array(x_data).astype(np.float32)
    y_data = np.array(y_data)
    y_data = to_categorical(y_data, num_classes=len(actions))
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=43)

    model = Sequential([
        LSTM(64, activation='relu', input_shape=x_train.shape[1:3]),
        Dense(32, activation='relu'),
        Dense(len(actions), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    callbacks = [
        ModelCheckpoint(model_path, monitor='val_acc', verbose=1, save_best_only=True, mode='auto'),
        ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=50, verbose=1, mode='auto'),
        EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=200,
        callbacks=callbacks
    )

    # 모델 학습 후 레이블 저장
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump(actions, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        gesture_name = sys.argv[1]
    else:
        gesture_name = input("학습할 제스처 이름을 입력하세요: ")
    data_path = './ai/training/data'
    model_path = './ai/models/gesture_model.h5'
    collect_and_train(gesture_name, data_path, model_path)
