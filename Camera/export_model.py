import os
import sys
from ultralytics import YOLO

def export_model(model_name="yolo11s-pose.pt"):
    print(f"\n=== 1. YOLO11 Pose 모델 '{model_name}' 다운로드 및 로드 ===")
    try:
        model = YOLO(model_name)
        print(f"모델 '{model_name}' 로드 성공!")
    except Exception as e:
        print(f"[오류] 모델 로드 실패: {e}")
        sys.exit(1)

    # 1. ONNX 변환
    print(f"\n=== 2. ONNX 포맷 변환 시작 ===")
    try:
        onnx_path = model.export(format="onnx")
        print(f"ONNX 변환 완료! 파일 경로: {onnx_path}")
    except Exception as e:
        print(f"[경고] ONNX 변환 중 에러 발생 (생략 가능): {e}")

    # 2. NCNN 변환 (라즈베리파이 최적화용)
    print(f"\n=== 3. NCNN 포맷 변환 시작 (라즈베리파이 CPU 용) ===")
    try:
        # ncnn 변환은 ultralytics 내부적으로 ncnn 패키지와 빌드 툴체인을 다운로드/사용합니다.
        ncnn_path = model.export(format="ncnn")
        print(f"NCNN 변환 완료! 폴더 경로: {ncnn_path}")
        print("\n🎉 모든 변환 작업이 성공적으로 완료되었습니다!")
        print(f"라즈베리파이에서 돌릴 때 '{ncnn_path}' 폴더를 모델명 대신 로컬에서 사용하세요.")
    except Exception as e:
        print(f"[경고] NCNN 변환 중 에러 발생: {e}")
        print("임베디드 NCNN 컴파일러 라이브러리 등이 설치되지 않았을 수 있습니다.")
        print("만약 윈도우 환경에서 실패했다면, 라즈베리파이 보드에서 이 스크립트를 직접 실행하시기를 권장합니다.")

if __name__ == "__main__":
    print("=" * 60)
    print("  YOLO11-Pose Raspberry Pi 가속 모델 변환기 (NCNN/ONNX)")
    print("=" * 60)
    
    # 인자로 모델명을 받거나 기본값 사용
    if len(sys.argv) > 1:
        selected_model = sys.argv[1]
    else:
        selected_model = "yolo11s-pose.pt"
        print(f"인자가 지정되지 않아 기본 모델 '{selected_model}'로 변환을 시작합니다.")
        print("사용법 예시: python export_model.py yolo11n-pose.pt")
        
    export_model(selected_model)
