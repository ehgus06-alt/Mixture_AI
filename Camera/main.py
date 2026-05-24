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
    # 라즈베리파이 등 Linux 환경에서 하드웨어 모델명 읽기
    try:
        with open('/sys/firmware/devicetree/base/model', 'r') as f:
            return f.read().strip('\x00').strip()
    except Exception:
        pass
    
    # Windows나 일반 PC의 경우 운영체제 정보나 호스트네임 사용
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
    #     if response.status_code != 200:
    #         print(f"[오류] 서버 응답 코드: {response.status_code}")
    # except Exception as e:
    #     print(f"[오류] 서버 전송 실패: {e}")

def main():
    # [YOLO11 Pose 업그레이드 및 NCNN 가속 지원]
    # 라즈베리파이에서 속도를 극대화하기 위해 NCNN 모델 폴더가 존재하면 먼저 로드하고,
    # 없으면 기본 PyTorch 모델(.pt)을 로드합니다.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ncnn_model_path = os.path.join(script_dir, "yolo11s-pose_ncnn_model")
    pt_model_path = os.path.join(script_dir, "yolo11s-pose.pt")
    
    if os.path.exists(ncnn_model_path):
        model_name = ncnn_model_path
        print(f"최적화된 NCNN 모델을 감지했습니다. NCNN 엔진으로 로드합니다: {model_name}")
    elif os.path.exists(pt_model_path):
        model_name = pt_model_path
        print(f"로컬 PyTorch 모델을 감지하여 로드합니다: {model_name}")
    else:
        # 파일이 없을 경우 ultralytics가 자동으로 기본 모델을 다운로드하도록 처리
        model_name = "yolo11s-pose.pt"
        print(f"YOLO11 Pose 모델을 불러오는 중입니다: {model_name}")
        print("💡 팁: 라즈베리파이에서 속도를 극대화하려면 'python export_model.py'를 실행해 NCNN 포맷으로 변환해 보세요!")
        
    model = YOLO(model_name)
    
    # 영상 소스 설정
    video_source = "hello.mp4"
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print("비디오 소스를 열 수 없습니다. 파일명이나 웹캠 연결을 확인해주세요.")
        return

    print("영상을 분석합니다. 종료하려면 'q' 키를 누르세요.")

    fall_frame_count = 0
    FALL_THRESHOLD = 3
    
    device_id = get_device_model()
    print(f"현재 기기 인식 결과: {device_id}")
    
    current_state = "UNKNOWN"
    
    # ID별 가로세로 비율(Aspect Ratio) 추적을 위한 딕셔너리
    ratio_history = defaultdict(list)
    HISTORY_LENGTH = 30 # 약 1초 (30fps 기준) 과거까지 기억

    while True:
        success, frame = cap.read()
        if not success:
            print("영상이 종료되었거나 프레임을 읽어올 수 없습니다.")
            break
            
        # 트래킹(Tracking) 기능 활성화: 사람마다 고유 ID 부여
        results = model.track(frame, conf=0.20, persist=True, verbose=False)
        
        fall_detected_in_current_frame = False
        annotated_frame = results[0].plot(boxes=False)
        
        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue
                
            # 트래킹 ID 파싱
            if boxes.id is not None:
                track_ids = boxes.id.int().cpu().tolist()
            else:
                track_ids = [None] * len(boxes)
                
            for i in range(len(boxes)):
                track_id = track_ids[i]
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                
                width = x2 - x1
                height = y2 - y1
                aspect_ratio = width / height if height > 0 else 0
                
                is_fallen = False
                
                if track_id is not None:
                    # 현재 프레임의 비율을 큐에 저장
                    ratio_history[track_id].append(aspect_ratio)
                    # 설정한 길이를 넘어가면 가장 오래된 과거 데이터 삭제
                    if len(ratio_history[track_id]) > HISTORY_LENGTH:
                        ratio_history[track_id].pop(0)
                        
                    # 과거와 현재 비교 로직 (최소 15프레임 이상 쌓였을 때 작동)
                    if len(ratio_history[track_id]) >= 15:
                        current_ratio = aspect_ratio
                        # 최근 10프레임을 제외한 순수 과거의 비율들을 추출
                        past_ratios = ratio_history[track_id][:-10] 
                        min_past_ratio = min(past_ratios) if len(past_ratios) > 0 else 1.0
                        
                        # [핵심 로직]
                        # 1. 현재 누워 있는가? (ratio > 1.2)
                        # 2. 1초 내에 확실히 서 있는 상태였는가? (ratio < 0.8)
                        if current_ratio > 1.2 and min_past_ratio < 0.8:
                            is_fallen = True
                else:
                    # ID 추적이 끊기거나 실패한 경우는 아주 극단적인 비율(1.5 초과)일 때만 잡음
                    if aspect_ratio > 1.5: 
                        is_fallen = True

                # 그리기 로직
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
        
        # 전체 화면 상태 판별 로직
        new_state = "UNKNOWN"
        if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
            new_state = "UNKNOWN"  # 아무도 인식되지 않은 상태
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
                new_state = current_state  # 상태 변화 중일 때는 이전 상태 유지

        # 시각적 경고 처리
        if new_state == "DANGER":
            cv2.putText(annotated_frame, "WARNING: FAST FALL DETECTED!!", (30, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4, cv2.LINE_AA)
            if fall_frame_count % 2 == 0:
                cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], annotated_frame.shape[0]), (0, 0, 255), 10)
                
        # 상태가 변경되었을 때만 JSON 페이로드 전송
        if current_state != new_state:
            # 향후 센서 모듈 상태를 이곳의 파라미터로 연결하시면 됩니다.
            send_event_payload(device_id, new_state, sensor_vibrator=True, sensor_radar=True, sensor_thermal=True)
            current_state = new_state

        cv2.imshow("Fall Detection - YOLOv8 Pose", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
