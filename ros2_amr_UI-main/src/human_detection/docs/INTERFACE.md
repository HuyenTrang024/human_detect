# Giao diện ROS 2 cho Web / Backend

Tài liệu này mô tả đầu ra của `gesture_node` để backend và giao diện web kết nối.
Cách cài đặt, build và chạy xem [../README.md](../README.md).

## 1. Tóm tắt

```
Camera ──/image_front──► gesture_node ──/gesture (Int32)──► Backend ──► Web
                              │
                              ├─ /gesture/name (String)    tên cử chỉ
                              ├─ /people/target_id (Int32) người chủ
                              └─ /gesture/debug_image      ảnh debug
```

- **Đầu vào:** `/image_front` (`sensor_msgs/Image`, bgr8) từ gói `camera`.
- **Đầu ra chính:** `/gesture` (`std_msgs/Int32`), số hiệu cử chỉ đã lọc.

## 2. Topic chính: `/gesture`

| Thuộc tính | Giá trị |
|---|---|
| Kiểu | `std_msgs/Int32` (đọc trường `data`) |
| Publisher QoS | Reliable, depth 10 |
| Khi nào publish | Mỗi frame ảnh được xử lý. Không có cử chỉ thì vẫn publish `5` |
| Đã lọc | Có (ngưỡng tin cậy, xác nhận nhiều lần, giữ lệnh, mất người) |

### Bảng số hiệu → cử chỉ

| Int32 | Tên (`/gesture/name`) | Ý nghĩa | `/cmd_vel` (nếu bật) |
|---|---|---|---|
| 0 | `MoveForward` | Đi tới | vx > 0 |
| 1 | `Halt` | Dừng | 0 |
| 2 | `MoveInReverse` | Đi lùi | vx = -0.15 |
| 3 | `Left` | Sang trái | 0 (chưa gắn chuyển động) |
| 4 | `Right` | Sang phải | 0 (chưa gắn chuyển động) |
| 5 | `None` | Không có lệnh / không đủ tin cậy | 0 |

Nguồn của bảng: `GESTURE_IDS` trong `human_detection/gesture/command_filter.py`.
Bảng này **cố định** và trùng thứ tự `CLASS_NAMES` lúc train model 5 lớp (đã bỏ `FollowMe`).
Thêm cử chỉ mới thì thêm vào cuối model và cuối bảng, rồi báo lại backend.

## 3. Các topic khác

| Topic | Kiểu | Nội dung |
|---|---|---|
| `/gesture/name` | `std_msgs/String` | Tên cử chỉ đã lọc, cùng nội dung với `/gesture` |
| `/gesture/raw` | `std_msgs/String` (JSON) | Dự đoán thô: `label`, `prob`, `person`, `skeleton`, `reason`, `probs` |
| `/gesture/person` | `vision_msgs/Detection2DArray` | Khung người đang theo dõi (pixel) |
| `/gesture/debug_image` | `sensor_msgs/Image` | Ảnh có khung người và `<cử chỉ> <xx>%`, dùng để hiển thị trên web |
| `/people/tracks` | `vision_msgs/Detection2DArray` | Mọi người trong khung, `id` = track_id |
| `/people/target_id` | `std_msgs/Int32` | track_id người chủ, `-1` = chưa khoá |
| `/people/target_state` | `std_msgs/String` | `IDLE` / `SEARCHING` / `LOCKED` / `LOST` |
| `/people/trajectories` | `std_msgs/String` (JSON) | Từ `trajectory_node`: vị trí, vận tốc, quỹ đạo dự đoán |
| `/cmd_vel` | `geometry_msgs/Twist` | Chỉ khi `enable_cmd_vel: true` (mặc định tắt) |

`/people/*` (ngoài `/people/trajectories`) chỉ có khi `enable_tracking: true`.
`/people/target_id` và `/people/target_state` chỉ có khi `enable_target_lock: true`.

## 4. Hành vi cần biết khi hiển thị

- **Bộ lọc:** lệnh chỉ đổi khi cùng một cử chỉ có xác suất ≥ `conf_threshold` (0.8) trong `confirm_count` (3) lần liên tiếp. `Halt` được nhận ngay.
- **Về `None` (5):** khi mất người quá `lost_sec` (1.0 s) hoặc không thấy cử chỉ quá `hold_sec` (3.0 s).
- **Độ trễ ban đầu:** cần khoảng 3,2 giây hình ảnh (cửa sổ 32 frame @ 10 FPS) mới có kết quả đầu tiên.
- **Khoá người chủ:** mặc định `enable_target_lock: false`, ai cũng điều khiển được.
  Khi bật, chưa khoá thì chỉ `Halt` đi qua, các cử chỉ khác thành `5`. Người làm `lock_gesture` (mặc định `Right`) sẽ được khoá.
  Web nên hiển thị `/people/target_state` kèm `/gesture` để người dùng hiểu vì sao robot chưa phản hồi.

## 5. Kiểm tra nhanh

```bash
ros2 topic list
ros2 topic echo /gesture            # data: 0..5
ros2 topic echo /gesture/name
ros2 topic hz /gesture/raw          # cần khoảng ≥ 10 Hz
ros2 topic info /gesture -v         # xem kiểu và QoS
```

## 6. Kết nối từ Web / Backend

### Cách A: rosbridge (web subscribe trực tiếp)

```bash
sudo apt install ros-jazzy-rosbridge-suite
ros2 launch rosbridge_server rosbridge_websocket_launch.xml   # ws://<robot_ip>:9090
```

```js
// roslibjs
const ros = new ROSLIB.Ros({ url: 'ws://<robot_ip>:9090' });

const GESTURES = ['MoveForward', 'Halt', 'MoveInReverse', 'Left', 'Right', 'None'];

new ROSLIB.Topic({ ros, name: '/gesture', messageType: 'std_msgs/Int32' })
  .subscribe(msg => console.log(msg.data, GESTURES[msg.data]));
```

### Cách B: backend dùng rclpy rồi đẩy xuống web (WebSocket/REST)

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32

GESTURES = ['MoveForward', 'Halt', 'MoveInReverse', 'Left', 'Right', 'None']

class GestureListener(Node):
    def __init__(self):
        super().__init__('gesture_listener')
        self.create_subscription(Int32, '/gesture', self.on_gesture, 10)

    def on_gesture(self, msg: Int32):
        name = GESTURES[msg.data] if 0 <= msg.data < len(GESTURES) else 'Unknown'
        self.get_logger().info(f'{msg.data} -> {name}')   # đẩy sang WebSocket ở đây

rclpy.init()
rclpy.spin(GestureListener())
```

Hiển thị ảnh debug: dùng `web_video_server` (`http://<robot_ip>:8080/stream?topic=/gesture/debug_image`)
hoặc subscribe `sensor_msgs/Image` qua rosbridge.

### Lưu ý mạng

Backend và robot phải cùng `ROS_DOMAIN_ID` và cùng mạng LAN. Khi test trên laptop một mình,
đặt `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST` để không lẫn với camera robot trong LAN.
