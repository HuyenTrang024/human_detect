# Human Detection – Nhận diện người và cử chỉ điều khiển robot AMR (ROS 2)

Dự án giúp robot AMR **nhận ra người trước camera và hiểu lệnh bằng cử chỉ tay**, chạy thời gian thực trên CPU (robot không có GPU) với **ROS 2 Jazzy**.

## Tính năng

- Phát hiện người bằng YOLOv5n (finetune 1 lớp `Person`).
- Lấy khung xương bằng MediaPipe Pose, phân loại cử chỉ bằng DD-Net (7 lớp).
- Theo dõi nhiều người bằng ByteTrack, khoá "người chủ" khi người đó làm cử chỉ `FollowMe`.
- Dự đoán quỹ đạo từng người (Kalman) để chuẩn bị cho phần tránh người.
- Công cụ quay dataset cử chỉ trực tiếp từ camera robot (`record_node`).

## Kiến trúc

```
/image_front ─► YOLOv5n ─► ByteTrack ─► Crop người ─► MediaPipe Pose ─► 15 khớp
                                                                          │
/gesture ◄─ CommandFilter ◄─ DD-Net ◄─ cửa sổ 32 frame @ 10 FPS (3,2 giây) ◄┘
```

## Mô hình và kết quả

| Thành phần | Nhiệm vụ | Huấn luyện | Kết quả |
|---|---|---|---|
| YOLOv5n | Tìm người | Finetune, 9.375 ảnh | mAP50 0,826 · mAP50-95 0,514 |
| MediaPipe Pose (lite) | Khung xương 33 điểm | Dùng sẵn | – |
| DD-Net (lite, 16 filters) | Phân loại cử chỉ | Train từ đầu trên RoCoG-v2 | Accuracy test 95,0% · Macro-F1 0,950 |
| ByteTrack | Gán ID người | Không có trọng số | – |

Chi tiết xem [`BAO_CAO_HUAN_LUYEN.md`](BAO_CAO_HUAN_LUYEN.md).

Bảy cử chỉ của model hiện tại (RoCoG-v2): Advance, Attention, Rally, MoveForward, Halt, FollowMe, MoveInReverse.
Bảng lớp mới (MoveForward, Halt, FollowMe, MoveInReverse, Left, Right, None) đã có trong code, nhưng model chưa được train lại.

## Cấu trúc thư mục

```
├── Human Detection/Human Detection/    Code huấn luyện: trích khung xương, train/đánh giá DD-Net, finetune YOLO
│   └── checkpoints/scratch_f16/        DD-Net đã train (trọng số, log, kết quả test)
├── ros2_amr_UI-main/src/
│   ├── camera/                         Package camera RealSense (C++)
│   └── human_detection/                Package ROS 2 nhận diện cử chỉ (Python)
├── install_ros2_jazzy.sh               Script cài ROS 2 Jazzy và thư viện
├── BAO_CAO_HUAN_LUYEN.md               Báo cáo huấn luyện mô hình
└── NHAT_KY.md                          Nhật ký quá trình thực hiện
```

Dữ liệu huấn luyện (`Dataset/`), kết quả YOLO (`runs/`) và khung xương đã trích (`keypoints/`) không nằm trong repo.

## Cài đặt

Yêu cầu: Ubuntu 24.04, ROS 2 Jazzy, Python 3.12.

```bash
./install_ros2_jazzy.sh                                     # cài ROS 2 và các gói hệ thống
./ros2_amr_UI-main/src/human_detection/setup_venv.sh        # tạo venv ~/gesture_venv (torch CPU, ultralytics, mediapipe...)
```

Lưu ý: giữ `numpy < 2` vì các gói ROS cài từ apt được build với numpy 1.x.

## Build và chạy

```bash
source /opt/ros/jazzy/setup.bash
source ~/gesture_venv/bin/activate
cd ros2_amr_UI-main
python3 -m colcon build --symlink-install
source install/setup.bash

ros2 launch human_detection gesture.launch.py           # nhận diện cử chỉ
ros2 launch human_detection people_tracking.launch.py   # thêm tracking và quỹ đạo
```

Cần node camera đang publish `/image_front`. Nếu chưa có RealSense, dùng `video_publisher` để phát video hoặc webcam lên topic này.

### Topic chính

| Topic | Nội dung |
|---|---|
| `/gesture` | Số hiệu lệnh đã lọc |
| `/gesture/name`, `/gesture/raw` | Tên lệnh, xác suất từng lớp (JSON) |
| `/gesture/person`, `/gesture/debug_image` | Khung người đang theo dõi, ảnh debug |
| `/people/tracks`, `/people/trajectories` | Mọi người có ID, quỹ đạo dự đoán |
| `/people/target_id`, `/people/target_state` | Người chủ đã khoá và trạng thái |
| `/cmd_vel` | Chỉ khi bật `enable_cmd_vel` (mặc định tắt) |

Bảng đầy đủ và tham số nằm trong [`ros2_amr_UI-main/src/human_detection/README.md`](ros2_amr_UI-main/src/human_detection/README.md) và `config/gesture_params.yaml`.

## Huấn luyện lại

```bash
cd "Human Detection/Human Detection"
python extract_keypoints.py --splits train val test --device 0       # video -> khung xương
python train_ddnet.py --mode scratch --name v2                       # train DD-Net
python evaluate_ddnet.py --ckpt checkpoints/v2/best.pt --split test  # đánh giá
```

Thu dữ liệu mới từ camera robot:

```bash
ros2 run human_detection record_node --ros-args -p subject:=S11 -p out_dir:=/duong/dan
```

## Hạn chế hiện tại

- Chưa kiểm thử với người thật trước camera robot; dữ liệu RoCoG-v2 quay trong điều kiện khác.
- Hai cặp cử chỉ dễ nhầm: `MoveForward` ↔ `MoveInReverse`, `Attention` ↔ `Rally`.
- Cử chỉ cần khoảng 3,2 giây để ra kết quả đầu tiên (cửa sổ 32 frame).
- Phần tracking mới được kiểm thử logic ngoài ROS; chưa có phần bám người và tránh người.

## Tài liệu tham khảo

- DD-Net: Yang et al., *Make Skeleton-based Action Recognition Model Smaller, Faster and Better*, 2019.
- RoCoG-v2: https://github.com/reddyav1/RoCoG-v2
- Ultralytics YOLO, MediaPipe Pose, ByteTrack.
