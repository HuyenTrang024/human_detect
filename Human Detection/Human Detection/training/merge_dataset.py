import os
import shutil
from pathlib import Path

# Danh sách đường dẫn tới 3 thư mục dataset gốc của bạn
# (Hãy thay thế bằng tên thực tế của 3 thư mục đó)
source_datasets = [
    Path("/media/itvkist/Projects/Hiep/Human Detection/Dataset/Human Dataset"),
    Path("/media/itvkist/Projects/Hiep/Human Detection/Dataset/person_1"),
    Path("/media/itvkist/Projects/Hiep/Human Detection/Dataset/person_2")
]

# Thư mục đích sau khi gộp chung
output_dataset = Path("Dataset/Merged_Human_Dataset")

# Các tập split thường có trong YOLO
splits = ["train", "valid", "test"]

for split in splits:
    # Tạo sẵn cấu trúc thư mục images/ và labels/ cho từng split
    (output_dataset / "images" / split).mkdir(parents=True, exist_ok=True)
    (output_dataset / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    for idx, src in enumerate(source_datasets):
        # Kiểm tra cấu trúc thư mục (thường là train/images hoặc images/train)
        # Dưới đây hỗ trợ cấu trúc phổ biến: folder_gốc/train/images
        img_dir = src / split / "images"
        lbl_dir = src / split / "labels"
        
        if not img_dir.exists():
            # Thử cách cấu trúc khác nếu có: folder_gốc/images/train
            img_dir = src / "images" / split
            lbl_dir = src / "labels" / split
            
        if not img_dir.exists():
            print(f"Bỏ qua vì không tìm thấy thư mục cho {split} trong {src}")
            continue
            
        print(f"Đang gộp {split} từ {src}...")
        
        for img_path in img_dir.glob("*.*"):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
                
            # Thêm tiền tố ds{idx}_ để tránh việc các dataset khác nhau có trùng tên ảnh (vd: img1.jpg)
            new_filename = f"ds{idx}_{img_path.name}"
            
            # 1. Copy file ảnh vào thư mục gộp chung
            shutil.copy(img_path, output_dataset / "images" / split / new_filename)
            
            # 2. Tìm file nhãn (.txt) tương ứng và copy qua với tên mới tương tự
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                new_lbl_name = Path(new_filename).stem + ".txt"
                shutil.copy(lbl_path, output_dataset / "labels" / split / new_lbl_name)

print("\n--- GỘP DATASET HOÀN TẤT! ---")
print(f"HÃY TẠO FILE data.yaml TRỎ VÀO THƯ MỤC: {output_dataset.resolve()}")