import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import json
from datetime import datetime, timezone
import platform
import os
import requests

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
    print(f"[{event_type} 상태 감지] 백엔드로 전송되는 JSON 데이터:")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"==================================================\n")
    
    # 서버가 없으므로 실제 전송 코드는 주석 처리해두었습니다. (불필요한 에러 로그 방지)
    # try:
    #     response = requests.post("http://your-backend-server.com/api/event", json=payload)
    # except Exception as e:
    #     pass

def main():
    print("[휴리스틱 기법] YOLO-Pose 기반 다이나믹 키포인트 추적 버전을 불러옵니다...")
    model = YOLO("yolov8m-pose.pt")
    
    video_source = "test.mp4"
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print("비디오 소스를 열 수 없습니다.")
        return

    fall_frame_count = 0
    FALL_THRESHOLD = 3
    
    device_id = get_device_model()
    current_state = "UNKNOWN"
    
    # ID별 이력 저장을 위한 딕셔너리 (aspect_ratio, center_y, height 저장)
    history = defaultdict(list)
    HISTORY_LENGTH = 30 # 약 1초 (30fps 기준) 과거까지 기억

    while True:
        success, frame = cap.read()
        if not success:
            break
            
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
            send_event_payload(device_id, new_state)
            current_state = new_state

        cv2.imshow("Heuristic Fall Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
