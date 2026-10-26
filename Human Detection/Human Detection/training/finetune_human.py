import argparse
import os

def train_detection(data_yaml, epochs=100, imgsz=640):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Vui lòng cài đặt thư viện: pip install ultralytics")
        return

    model = YOLO("yolov5n.pt") 
    
    print(f"--- BẮT ĐẦU FINETUNE YOLOv5n DETECTION ---")
    
    # Bắt đầu quá trình huấn luyện
    model.train(
        data=data_yaml, 
        epochs=epochs, 
        imgsz=imgsz, 
        batch=16, 
        project="runs/detect", 
        name="robot_person_detect",
        lr0=0.01,              # Learning rate ban đầu
        mosaic=1.0,            # Ghép 4 ảnh thành 1 để học context tốt hơn
        degrees=10,            # Xoay ảnh tối đa 10 độ
        translate=0.1,         # Dịch chuyển ảnh
        fliplr=0.5,            # Lật ngang 50%
        patience=20,           # Early stopping: Dừng nếu 20 epoch không giảm loss
        exist_ok=True,
        workers=8
    )
    
    # Đánh giá lại mô hình trên tập validation
    print("\n--- ĐÁNH GIÁ MÔ HÌNH ---")
    model.val()

def export_model(weights_path, fmt="onnx", imgsz=640):
    from ultralytics import YOLO

    if not os.path.exists(weights_path):
        print(f"Lỗi: Không tìm thấy file trọng số tại {weights_path}")
        return

    print(f"\n--- XUẤT MÔ HÌNH SANG {fmt.upper()} ---")
    model = YOLO(weights_path)
    
    # half=True để dùng FP16 (giảm một nửa dung lượng/RAM, tăng tốc inference trên GPU/NPU)
    model.export(
        format=fmt, 
        imgsz=imgsz, 
        half=(fmt == "engine"), 
        simplify=True
    )

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "export"], help="Lệnh: train hoặc export")
    ap.add_argument("--data", default="Dataset/Merged_Human_Dataset/data.yaml", help="Đường dẫn file cấu hình YAML")
    ap.add_argument("--weights", default="runs/detect/runs/detect/robot_person_detect/weights/best.pt", help="File trọng số để export")
    ap.add_argument("--format", default="onnx", choices=["onnx", "engine", "tflite", "openvino"], help="Định dạng xuất")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh train (thường dùng 640 cho detection)")
    
    args = ap.parse_args()
    
    if args.cmd == "train":
        train_detection(args.data, epochs=args.epochs, imgsz=args.imgsz)
    elif args.cmd == "export":
        export_model(args.weights, fmt=args.format, imgsz=args.imgsz)