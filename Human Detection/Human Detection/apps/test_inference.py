import cv2
import random
from pathlib import Path
from ultralytics import YOLO

# 1. Đường dẫn tới file weights tốt nhất sau khi train
weights_path = "runs/detect/runs/detect/robot_person_detect/weights/best.pt"

# 2. Đường dẫn tới thư mục chứa ảnh test của bạn
test_img_dir = Path("Dataset/Merged_Human_Dataset/images/test")

if not Path(weights_path).exists():
    print(f"Không tìm thấy file weights tại: {weights_path}. Hãy kiểm tra lại!")
    exit()

# Lấy danh sách toàn bộ ảnh trong tập test
all_test_imgs = list(test_img_dir.glob("*.jpg")) + list(test_img_dir.glob("*.png"))
if not all_test_imgs:
    print(f"Không tìm thấy ảnh nào trong thư mục: {test_img_dir}")
    exit()

# Chọn ngẫu nhiên 1 ảnh để test
selected_img = random.choice(all_test_imgs)
print(f"Đang dự đoán trên ảnh: {selected_img}")

# 3. Load mô hình YOLO đã train
model = YOLO(weights_path)

# 4. Chạy dự đoán (Inference) trên ảnh với độ tin cậy conf=0.4
results = model(str(selected_img), conf=0.4)

# 5. Hiển thị kết quả trực tiếp (không lưu)
for r in results:
    # r.plot() sẽ tự động vẽ bounding box và nhãn lên ảnh
    im_bgr = r.plot()  # Trả về mảng ảnh BGR (OpenCV format)
    
    # Hiển thị cửa sổ ảnh trực tiếp
    cv2.imshow("YOLO Detection Result", im_bgr)
    cv2.waitKey(0)  # Nhấn phím bất kỳ trên bàn phím để đóng ảnh
    cv2.destroyAllWindows()