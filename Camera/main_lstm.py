import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import json
from datetime import datetime, timezone
import platform
import os
import torch
import torch.nn as nn
import requests

# ---------------------------------------------------------
# [1] 시퀀스 분류용 LSTM 딥러닝 모델 정의 (PyTorch)
# ---------------------------------------------------------
class FallDetectionLSTM(nn.Module):
    def __init__(self, input_size=34, hidden_size=64, num_layers=2, num_classes=2):
        super(FallDetectionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM 레이어: 17개 관절(x, y) = 34차원 입력
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Fully Connected 분류 레이어
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x 형태: (batch_size, sequence_length, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        # 마지막 시퀀스의 출력만 사용하여 넘어짐 여부 분류
        out = self.fc(out[:, -1, :])
        return out

# ---------------------------------------------------------
# [2] 디바이스 정보 및 통신 헬퍼 함수
# ---------------------------------------------------------
def get_device_model():
    try:
        with open('/sys/firmware/devicetree/base/model', 'r') as f:
            return f.read().strip('\x00').strip()
    except Exception:
        pass
    
    os_name = platform.system()
    if os_name == "Windows":
        return f"Windows_PC_{platform.node()}"
    elif os_name == "Linux":
        return f"Linux_PC_{platform.node()}"
    elif os_name == "Darwin":
        return f"Mac_{platform.node()}"
    return "Unknown_Device"

def send_event_payload(device_id, event_type, sensor_vibrator=True, sensor_radar=True, sensor_thermal=True):
    payload = {
        "device_id": device_id,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "event_type": event_type,
        "sensor_status": {                    
            "vibrator": sensor_vibrator,
            "radar": sensor_radar,
            "thermal_imaging": sensor_thermal
        }
    }
    
    print(f"\n==================================================")
    print(f"[{event_type} 상태 감지 (LSTM)] 백엔드로 전송되는 JSON 데이터:")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"==================================================\n")

# ---------------------------------------------------------
# [3] 메인 실행 함수
# ---------------------------------------------------------
def main():
    print("[딥러닝 기법] YOLO-Pose + LSTM 추론 파이프라인을 불러옵니다...")
    
    # 1. YOLO 모델 로드 (Feature Extractor)
    yolo_model = YOLO("yolov8m-pose.pt")
    
    # 2. LSTM 신경망 모델 로드 (Classifier)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lstm_model = FallDetectionLSTM().to(device)
    
    # [중요] 미리 학습된 가중치 파일(fall_lstm.pth)이 반드시 필요합니다!
    weights_path = "fall_lstm.pth"
    if os.path.exists(weights_path):
        lstm_model.load_state_dict(torch.load(weights_path, map_location=device))
        lstm_model.eval()
        print(f"[{weights_path}] LSTM 모델 가중치를 성공적으로 불러왔습니다.")
    else:
        print("\n[에러] 사전 학습된 LSTM 가중치 파일(fall_lstm.pth)을 찾을 수 없습니다!")
        print("이 코드가 작동하려면 데이터셋을 수집하여 LSTM을 학습시키는 과정이 선행되어야 합니다.")
        print("당장은 동작을 확인하실 수 없으니, main_heuristic.py 를 먼저 사용해보시기를 권장합니다.\n")
        # 실제 환경에서는 여기서 return 처리하여 종료하지만, UI 구경을 위해 강행할 수도 있습니다.
        # return
    
    video_source = "hello.mp4"
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("비디오 소스를 열 수 없습니다.")
        return

    device_id = get_device_model()
    current_state = "UNKNOWN"
    
    # LSTM에는 지정된 길이(Sequence Length)의 연속된 데이터가 들어가야 함 (예: 30프레임 = 1초)
    SEQ_LENGTH = 30 
    
    # ID별 관절 좌표 시퀀스 저장
    # shape: (30, 34) (30프레임 동안의 17개 관절 x,y 좌표)
    pose_sequences = defaultdict(list)
    
    fall_frame_count = 0
    FALL_THRESHOLD = 3

    while True:
        success, frame = cap.read()
        if not success:
            break
            
        results = yolo_model.track(frame, conf=0.20, persist=True, verbose=False)
        annotated_frame = results[0].plot(boxes=False)
        fall_detected_in_current_frame = False
        
        for r in results:
            boxes = r.boxes
            keypoints = r.keypoints
            
            if boxes is None or len(boxes) == 0 or keypoints is None:
                continue
                
            track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)
                
            for i in range(len(boxes)):
                track_id = track_ids[i]
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                
                if track_id is not None and len(keypoints.xy) > i:
                    kp = keypoints.xy[i].cpu().numpy() # shape (17, 2)
                    
                    # 17개 관절 좌표를 1차원 배열(34,)로 변환 (정규화 필요 시 추가 연산)
                    flat_kp = kp.flatten()
                    
                    pose_sequences[track_id].append(flat_kp)
                    
                    # 큐 길이가 SEQ_LENGTH를 초과하면 오래된 데이터 삭제
                    if len(pose_sequences[track_id]) > SEQ_LENGTH:
                        pose_sequences[track_id].pop(0)
                        
                    is_fallen = False
                    
                    # 시퀀스가 꽉 찼을 때만 딥러닝 추론 (Inference)
                    if len(pose_sequences[track_id]) == SEQ_LENGTH:
                        # (1, 30, 34) 형태로 Tensor 변환
                        seq_tensor = torch.tensor(np.array([pose_sequences[track_id]]), dtype=torch.float32).to(device)
                        
                        # 모델 추론
                        with torch.no_grad():
                            output = lstm_model(seq_tensor)
                            probability = torch.softmax(output, dim=1)
                            # 인덱스 1이 낙상(Fall) 클래스라고 가정
                            fall_prob = probability[0][1].item()
                            
                            if fall_prob > 0.8: # 임계값 80% 이상 낙상 확률일 때
                                is_fallen = True
                    
                    if is_fallen:
                        fall_detected_in_current_frame = True
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
                        label = f"ID:{track_id} FALLEN(LSTM)!"
                        cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3, cv2.LINE_AA)
                    else:
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(annotated_frame, "STANDING", (int(x1), int(y1) - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        
        # 상태 전환 로직
        new_state = "UNKNOWN"
        if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
            new_state = "UNKNOWN"
        else:
            if fall_detected_in_current_frame:
                fall_frame_count += 1
            else:
                fall_frame_count = max(0, fall_frame_count - 1)
                
            if fall_frame_count >= FALL_THRESHOLD:
                new_state = "DANGER"
            elif fall_frame_count == 0:
                new_state = "SAFE"
            else:
                new_state = current_state 

        if new_state == "DANGER":
            cv2.putText(annotated_frame, "WARNING: DEEP FALL DETECTED!!", (30, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4, cv2.LINE_AA)
            if fall_frame_count % 2 == 0:
                cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], annotated_frame.shape[0]), (0, 0, 255), 10)
                
        if current_state != new_state:
            send_event_payload(device_id, new_state)
            current_state = new_state

        cv2.imshow("Deep Learning Fall Detection (LSTM)", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
