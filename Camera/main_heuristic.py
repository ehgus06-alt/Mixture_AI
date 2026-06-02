import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import json
from datetime import datetime, timezone
import platform
import os
import requests
import uuid
import socket
import psutil

def get_mac_address():
    try:
        # 활성화된 인터넷 연결의 로컬 IP를 가져옴
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        
        # 해당 IP가 할당된 네트워크 어댑터의 MAC 주소를 추출
        for interface, addrs in psutil.net_if_addrs().items():
            if any(addr.address == local_ip for addr in addrs):
                for addr in addrs:
                    if ('-' in addr.address or ':' in addr.address) and len(addr.address) == 17:
                        return addr.address.replace('-', ':').upper()
    except Exception:
        pass
        
    # 실패할 경우 기존 방식으로 Fallback
    mac = uuid.getnode()
    return ':'.join(('%012X' % mac)[i:i+2] for i in range(0, 12, 2))

def send_event_payload(device_id, event_type, sensor_vibrator=False, sensor_radar=False, sensor_thermal=False):
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
    print(f"[{event_type} 상태 감지] 백엔드로 전송되는 JSON 데이터:")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"==================================================\n")
    
     #서버가 없으므로 실제 전송 코드는 주석 처리해두었습니다. (불필요한 에러 로그 방지)
    try:
        response = requests.post("http://192.168.1.154:8080/api/device/data", json=payload)
    except Exception as e:
        pass

def main():
    print("[휴리스틱 기법] YOLO11-Pose 기반 다이나믹 키포인트 추적 버전을 불러옵니다...")
    
    # [YOLO11 Pose 업그레이드 및 NCNN/ONNX 가속 지원]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ncnn_model_path = os.path.join(script_dir, "yolo11s-pose_ncnn_model")
    onnx_model_path = os.path.join(script_dir, "yolo11s-pose.onnx")
    pt_model_path = os.path.join(script_dir, "yolo11s-pose.pt")
    
    if os.path.exists(ncnn_model_path):
        model_name = ncnn_model_path
        print(f"최적화된 NCNN 모델을 감지했습니다. NCNN 엔진으로 로드합니다: {model_name}")
    elif os.path.exists(onnx_model_path):
        model_name = onnx_model_path
        print(f"ONNX 모델을 감지했습니다. ONNX 엔진으로 로드합니다: {model_name}")
    elif os.path.exists(pt_model_path):
        model_name = pt_model_path
        print(f"로컬 PyTorch 모델을 감지하여 로드합니다: {model_name}")
    else:
        # 파일이 없을 경우 ultralytics가 자동으로 기본 모델을 다운로드하도록 처리
        model_name = "yolo11s-pose.pt"
        print(f"YOLO11 Pose 모델을 불러오는 중입니다: {model_name}")
        print("💡 팁: 라즈베리파이에서 속도를 극대화하려면 'python export_model.py'를 실행해 NCNN/ONNX 포맷으로 변환해 보세요!")
        
    model = YOLO(model_name)
    
    # [수정] Lepton 3.0 카메라 인덱스 설정 (UVC 캡처 보드 또는 V4L2 드라이버 기준 보통 0번)
    video_source = 0
    cap = cv2.VideoCapture(video_source)
    
    # Lepton 3.0 고유 해상도 (160x120) 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
    
    if not cap.isOpened():
        print(f"[{video_source}]번 카메라(Lepton) 소스를 열 수 없습니다.")
        return

    fall_frame_count = 0
    FALL_THRESHOLD = 3
    
    device_id = get_mac_address()
    current_state = "UNKNOWN"
    
    # ID별 이력 저장을 위한 딕셔너리 (aspect_ratio, center_y, height 저장)
    history = defaultdict(list)
    HISTORY_LENGTH = 30 # 약 1초 (30fps 기준) 과거까지 기억

    while True:
        success, frame = cap.read()
        if not success:
            print("카메라에서 프레임을 읽어올 수 없습니다.")
            break
            
        # [추가] 열화상 카메라 프레임 전처리
        # 프레임이 1채널(흑백)일 경우 YOLO 처리를 위해 3채널(BGR)로 변환
        if len(frame.shape) == 2 or frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
        # 160x120 해상도는 너무 작아 객체 인식 및 결과 확인이 어려우므로 640x480으로 확대
        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
            
        results = model.track(frame, conf=0.20, persist=True, verbose=False)
        
        fall_detected_in_current_frame = False
        annotated_frame = results[0].plot(boxes=False)
        
        for r in results:
            boxes = r.boxes
            keypoints = r.keypoints
            if boxes is None or len(boxes) == 0:
                continue
                
            track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)
                
            for i in range(len(boxes)):
                track_id = track_ids[i]
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                
                width = x2 - x1
                height = y2 - y1
                aspect_ratio = width / height if height > 0 else 0
                
                # Pose Keypoint에서 어깨(Shoulder) 또는 골반(Hip)의 Y좌표(높이) 추출
                center_y = (y1 + y2) / 2 # 기본 중심값
                if keypoints is not None and len(keypoints.xy) > i:
                    kp = keypoints.xy[i].cpu().numpy()
                    if len(kp) > 6:
                        # 5: 왼쪽 어깨, 6: 오른쪽 어깨
                        shoulder_y = (kp[5][1] + kp[6][1]) / 2 if kp[5][1] > 0 and kp[6][1] > 0 else 0
                        if shoulder_y > 0:
                            center_y = shoulder_y
                
                is_fallen = False
                
                if track_id is not None:
                    history[track_id].append((aspect_ratio, center_y, height))
                    if len(history[track_id]) > HISTORY_LENGTH:
                        history[track_id].pop(0)
                        
                    if len(history[track_id]) >= 15:
                        current_ratio, current_y, current_h = history[track_id][-1]
                        past_data = history[track_id][:-10] 
                        
                        min_past_ratio = min(d[0] for d in past_data)
                        avg_past_y = sum(d[1] for d in past_data) / len(past_data)
                        
                        # [핵심 1] 1초 전 대비 Y좌표가 확 떨어졌는가? (키의 35% 이상 하강)
                        y_drop_distance = current_y - avg_past_y
                        is_drop_fast = y_drop_distance > (current_h * 0.35) 
                        
                        # [핵심 2] 현재 누워있는 형태인가? (가로로 김)
                        is_lying_down = current_ratio > 1.0
                        
                        # [핵심 3] 과거에는 서 있었는가?
                        was_standing = min_past_ratio < 0.9
                        
                        # 세 가지 조건이 모두 충족되어야만 확실한 낙상(DANGER)으로 감지!
                        # (바닥에 원래 누워있던 사람이 스트레칭하는 경우 등을 걸러냄)
                        if is_drop_fast and is_lying_down and was_standing:
                            is_fallen = True
                else:
                    if aspect_ratio > 1.5: 
                        is_fallen = True

                if is_fallen:
                    fall_detected_in_current_frame = True
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
                    label = f"ID:{track_id} FALLEN!" if track_id else "FALLEN!"
                    cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3, cv2.LINE_AA)
                else:
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    label = f"ID:{track_id} STANDING" if track_id else "STANDING"
                    cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        
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
            cv2.putText(annotated_frame, "WARNING: FAST FALL DETECTED!!", (30, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4, cv2.LINE_AA)
            if fall_frame_count % 2 == 0:
                cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], annotated_frame.shape[0]), (0, 0, 255), 10)
                
        if current_state != new_state:
            # 상태가 UNKNOWN이 아니면(즉, SAFE나 DANGER면) 열화상 카메라가 객체를 인식한 것으로 간주
            is_thermal_detected = (new_state != "UNKNOWN")
            send_event_payload(device_id, new_state, sensor_thermal=is_thermal_detected)
            current_state = new_state

        cv2.imshow("Heuristic Fall Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
