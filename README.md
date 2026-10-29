# Human Detection

Robot AMR nhận ra người trước camera và hiểu lệnh bằng cử chỉ tay. Chạy trên CPU, ROS 2 Jazzy.

## Cách hoạt động

```
Camera ─► YOLOv5n ─► ByteTrack ─► MediaPipe Pose ─► DD-Net ─► lệnh (/gesture)
          tìm người   gán ID      khung xương      phân loại
                                                   cử chỉ
```

Năm cử chỉ: `MoveForward`, `Halt`, `MoveInReverse`, `Left`, `Right`.

| Bước | Mô hình | Đầu vào → đầu ra | Kết quả |
|---|---|---|---|
| Tìm người | YOLOv5n (finetune) | ảnh camera → bbox người | Test 1.044 ảnh: mAP50 **0,859**, mAP50-95 0,537 |
| Theo dõi người | ByteTrack | bbox → ID ổn định | – |
| Khung xương | MediaPipe Pose lite (dùng sẵn) | ảnh crop một người → 33 điểm → 15 khớp | 99,96% frame có khung xương dùng được (314 video) |
| Phân loại cử chỉ | DD-Net lite (train từ đầu, 5 lớp) | 32 frame × 15 khớp × (x, y) → 5 lớp | Test 105 video: **100%**, Macro-F1 1,000; validation 98,1% |

Test DD-Net chỉ có 105 video và 6 người, chưa thử với người thật trước camera robot, nên 100% chưa đảm bảo cho điều kiện thực tế.

Tài liệu:
- [KET_QUA_HUAN_LUYEN.md](KET_QUA_HUAN_LUYEN.md): kết quả từng mô hình, đầu vào/đầu ra từng khối
- [BAO_CAO_HUAN_LUYEN.md](BAO_CAO_HUAN_LUYEN.md): báo cáo huấn luyện đầy đủ
- [DATASET_ROBOT.md](DATASET_ROBOT.md): dataset (robot + RoCoG), ID người và số video theo train/validation/test
- [NHAT_KY.md](NHAT_KY.md): nhật ký làm việc

## Cấu trúc thư mục

```
human_detect/
│
├── Human Detection/Human Detection/   HUẤN LUYỆN MÔ HÌNH
│   ├── apps/                             demo camera, detect_person, robot_control
│   ├── gesture/                          DD-Net, pose, nhận diện cử chỉ
│   ├── training/                         extract_keypoints, train/evaluate DD-Net, finetune YOLO
│   ├── checkpoints/                      DD-Net đã train + kết quả (v3_5cls là bản đang dùng)
│   └── data/keypoints_v3/                keypoint của 316 video (không có trong repo)
│
├── Human Detection/Dataset/           DỮ LIỆU
│   ├── Dataset_Used/                     316 video model dùng: train 158, validation 53, test 105
│   │                                     (real/{ground,robot}/<Lớp>/ + annotations/all_{train,val,test}.txt)
│   ├── real_ground/                      bản đầy đủ: video robot, RoCoG, rejected/, file chia gốc
│   └── Merged_Human_Dataset/             9.375 ảnh người để finetune YOLO
│
├── ros2_amr_UI-main/src/              ROS 2 WORKSPACE
│   ├── camera/                           đọc camera RealSense
│   └── human_detection/                  nhận diện cử chỉ (package chính)
│       ├── human_detection/                 code: gesture_node, trajectory_node, record_node
│       ├── config/                          tham số (yaml)
│       ├── launch/                          file launch
│       └── models/                          best.pt, ddnet_gesture.pt, pose_landmarker_lite.task
│
├── install_ros2_jazzy.sh              cài ROS 2
├── BAO_CAO_HUAN_LUYEN.md              báo cáo huấn luyện
├── KET_QUA_HUAN_LUYEN.md              kết quả YOLO, MediaPipe, DD-Net
├── DATASET_ROBOT.md                   mô tả dataset
└── NHAT_KY.md                         nhật ký làm việc
```

Dữ liệu train (`Dataset/`, `runs/`, `data/keypoints*/`) không có trong repo.

## Cài đặt và chạy

```bash
./install_ros2_jazzy.sh
./ros2_amr_UI-main/src/human_detection/setup_venv.sh     # tạo venv ~/gesture_venv

source /opt/ros/jazzy/setup.bash
source ~/gesture_venv/bin/activate
cd ros2_amr_UI-main
python3 -m colcon build --symlink-install
source install/setup.bash

ros2 launch human_detection gesture.launch.py             # nhận diện cử chỉ
ros2 launch human_detection people_tracking.launch.py     # thêm theo dõi người + quỹ đạo
```

Cần camera publish `/image_front`. Chưa có camera thì dùng `video_publisher` để phát video.

## Topic chính

| Topic | Nội dung |
|---|---|
| `/gesture` | Lệnh đã lọc (số hiệu) |
| `/gesture/raw` | Xác suất từng cử chỉ (JSON) |
| `/people/tracks` | Mọi người trong khung, có ID |
| `/people/target_id` | ID của người chủ đã khoá |
| `/people/trajectories` | Quỹ đạo dự đoán |

Danh sách đầy đủ: [package README](ros2_amr_UI-main/src/human_detection/README.md).

## Huấn luyện lại

Chạy từ `Human Detection/Human Detection/`, dùng bộ video đã xếp sẵn trong `Dataset_Used`:

```bash
cd "Human Detection/Human Detection"
A=../Dataset/Dataset_Used
python -m training.rocog_labels --ann-dir $A/annotations --dataset-root $A \
    --splits all_train all_val all_test --check-files          # kiểm tra: 158 / 53 / 105 video, 316/316 tìm thấy
python -m training.extract_keypoints --splits all_train all_val all_test --device 0 \
    --ann-dir $A/annotations --dataset-root $A --out-dir data/keypoints_v3
python -m training.train_ddnet --mode scratch --train-splits all_train --val-split all_val \
    --ann-dir $A/annotations --kp-dir data/keypoints_v3 --name v3_5cls --device cuda
python -m training.evaluate_ddnet --ckpt checkpoints/v3_5cls/best.pt --split all_test \
    --ann-dir $A/annotations --kp-dir data/keypoints_v3
```

Mặc định `KEYPOINT_DIR` trong `gesture/config.py` là `data/keypoints_v3` (từ 29/10); `data/keypoints` là bản 7 lớp cũ.

## Hạn chế

- Chưa thử với người thật trước camera robot; test chỉ có 105 video (6 người), 77 video quay cùng thiết bị với train/val.
- `MoveForward` chỉ có data robot từ S11, S12 trong train (S13, S14 không có); đây là lớp duy nhất có lỗi ở validation.
- Model không có lớp `None`: robot lọc dự đoán yếu bằng ngưỡng `conf`, chưa đo tỉ lệ báo nhầm.
- Chưa đo lại FPS/CPU của cả pipeline với model 5 lớp.
- Cử chỉ cần khoảng 3,2 giây để ra kết quả.
- Chưa có phần bám người và tránh người.
