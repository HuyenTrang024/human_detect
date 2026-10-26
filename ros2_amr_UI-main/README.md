# ros2_amr_UI

Workspace ROS 2 (Jazzy) cho robot AMR đi theo người bằng cử chỉ.

```
ros2_amr_UI-main/
└── src/
    ├── camera/            (C++)    Đọc camera Intel RealSense, publish /image_front, /image_rear
    └── human_detection/   (Python) Nhận diện người, cử chỉ, theo dõi và dự đoán quỹ đạo
        ├── human_detection/
        │   ├── nodes/       Các node ROS: gesture_node, trajectory_node, record_node, video_publisher
        │   ├── gesture/     Pipeline cử chỉ: YOLO → MediaPipe Pose → DD-Net → bộ lọc lệnh
        │   ├── tracking/    ByteTrack, Kalman, chọn người chủ, dự đoán quỹ đạo
        │   ├── utils/       Hàm dùng chung (chuyển ảnh ROS ↔ numpy)
        │   └── tools/       Công cụ chạy không cần ROS (tracking_demo)
        ├── launch/          gesture.launch.py, people_tracking.launch.py
        ├── config/          Tham số (*.yaml)
        ├── models/          Trọng số: best.pt (YOLO), ddnet_gesture.pt, pose_landmarker_lite.task
        ├── scripts/         setup_venv.sh (tạo môi trường ảo cài thư viện Python)
        └── docs/            INTERFACE.md: topic và bảng số hiệu cử chỉ cho web/backend
```

## Luồng dữ liệu

```
camera ──/image_front──► gesture_node ──/gesture (Int32)──► web / backend
                              └──/people/tracks──► trajectory_node ──/people/trajectories
```

## Bắt đầu nhanh

```bash
cd src/human_detection && ./scripts/setup_venv.sh          # một lần
source /opt/ros/jazzy/setup.bash && source ~/gesture_venv/bin/activate
cd <ros2_ws> && python3 -m colcon build --symlink-install && source install/setup.bash
ros2 launch camera camera.launch.py
ros2 launch human_detection gesture.launch.py
```

Chi tiết: [src/human_detection/README.md](src/human_detection/README.md).
Topic và bảng số hiệu cử chỉ cho web: [src/human_detection/docs/INTERFACE.md](src/human_detection/docs/INTERFACE.md).
Mã huấn luyện model nằm ngoài workspace này, trong thư mục `Human Detection/Human Detection/`.
