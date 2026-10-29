# BÁO CÁO CÔNG VIỆC: HUẤN LUYỆN MÔ HÌNH NHẬN DIỆN NGƯỜI VÀ CỬ CHỈ CHO ROBOT AMR

**Dự án:** Human Detection trên ROS 2 (điều khiển robot AMR bằng cử chỉ)
**Thời gian:** 16/10/2026 – 29/10/2026
**Người thực hiện:** Trang To

> **Cập nhật 29/10/2026:** báo cáo đã chuyển từ model DD-Net 7 lớp RoCoG cũ (`scratch_f16`, test 95,0%) sang **model 5 lớp `v3_5cls`**
> (MoveForward, Halt, MoveInReverse, Left, Right) train bằng RoCoG + video quay bằng camera robot.
> Số liệu chi tiết từng khối và đầu vào/đầu ra: [KET_QUA_HUAN_LUYEN.md](KET_QUA_HUAN_LUYEN.md). Dữ liệu: [DATASET_ROBOT.md](DATASET_ROBOT.md). Nhật ký: [NHAT_KY.md](NHAT_KY.md).

---

## 1. Tóm tắt

Mục tiêu: robot AMR nhận ra người đứng trước camera và hiểu lệnh bằng cử chỉ tay, chạy thời gian thực trên CPU (robot không có GPU) trong ROS 2 Jazzy.

Hệ thống ghép **bốn thành phần**. Hai thành phần có huấn luyện riêng, hai thành phần dùng sẵn hoặc không có trọng số:

| Thành phần | Nhiệm vụ | Huấn luyện? | Kết quả chính |
|---|---|---|---|
| **YOLOv5n** (finetune) | Tìm người trong ảnh | Có: finetune 1 lớp `Person` | Test (1.044 ảnh): mAP50 **0,859**, mAP50-95 **0,537**; validation mAP50 0,826 |
| **MediaPipe Pose** (lite) | Lấy khung xương 33 điểm | Không: model có sẵn | Trên 314 video: 99,96% frame có khung xương dùng được |
| **DD-Net** (lite, 16 filters) | Phân loại cử chỉ từ chuỗi khung xương | Có: train từ đầu | Test (105 video): độ chính xác **100%**, Macro-F1 **1,000**; validation 98,1% |
| **ByteTrack** | Gán ID ổn định cho từng người | Không: thuật toán | – |

**Kết quả đạt được:** pipeline hoàn chỉnh chạy được trong ROS 2 trên CPU. Model cử chỉ hiện tại (`v3_5cls`) đã được chép vào
`models/ddnet_gesture.pt` của package ROS.

---

## 2. Kiến trúc tổng thể

```
Ảnh camera ─► YOLOv5n ─► ByteTrack ─► Cắt ảnh người ─► MediaPipe Pose ─► Chuẩn hoá khung xương
 (/image_front)  (tìm người)  (gán ID)   (crop + làm mượt)  (33 điểm)         (15 khớp)

   ─► cửa sổ 32 frame @10 FPS (3,2 giây) ─► DD-Net ─► 5 cử chỉ ─► CommandFilter ─► /gesture
```

Năm cử chỉ của model: `MoveForward`, `Halt`, `MoveInReverse`, `Left`, `Right`. Model không có lớp `None`; robot lọc dự đoán yếu bằng ngưỡng `conf` trong `command_filter.py`.

Bảng đầu vào/đầu ra của từng khối: xem mục 1 của [KET_QUA_HUAN_LUYEN.md](KET_QUA_HUAN_LUYEN.md).

Lý do chọn hướng "phát hiện người → khung xương → phân loại chuỗi" thay vì phân loại trực tiếp từ video:
- Khung xương nhẹ, chạy được trên CPU.
- DD-Net chỉ có phần phân loại cuối, nên model rất nhỏ (0,62 MB, 145.701 tham số, ~1,4 ms mỗi cửa sổ trên 1 luồng CPU).
- Đặc trưng khoảng cách giữa các khớp không phụ thuộc vị trí hay kích thước người trong ảnh.

---

## 3. Mô hình 1: YOLOv5n – phát hiện người

### 3.1. Vai trò
Tìm người trong từng frame, cho khung bao (bbox) để cắt ảnh đưa vào bước lấy khung xương và để ByteTrack theo dõi.

### 3.2. Dữ liệu
Gộp 3 bộ dữ liệu người (định dạng Roboflow/YOLO, 1 lớp `Person`) thành `Merged_Human_Dataset`:

| Tập | Số ảnh |
|---|---|
| Train | 6.288 |
| Validation | 2.043 |
| Test | 1.044 |
| **Tổng** | **9.375** |

Nguồn: `Human Dataset` (2.188 ảnh), `person_1` (1.484 ảnh), `person_2` (5.703 ảnh).

### 3.3. Cấu hình huấn luyện
| Tham số | Giá trị |
|---|---|
| Model khởi tạo | `yolov5n.pt` (pretrained COCO) |
| Số epoch tối đa / patience | 60 / 20 (early stopping) |
| Kích thước ảnh / batch | 640 / 16 |
| Optimizer / lr khởi đầu | auto / 0,01 |
| Số epoch thực tế chạy | 55 (~32 phút); `best.pt` ở epoch 35 |
| Công cụ | Ultralytics (`finetune_human.py`) |

### 3.4. Kết quả

| Tập | Precision | Recall | mAP50 | mAP50-95 | Nguồn |
|---|---|---|---|---|---|
| Validation, epoch 35 (`best.pt`) | 0,856 | 0,721 | 0,826 | 0,514 | `results.csv` khi train |
| Validation, đo lại 29/10 | 0,856 | 0,721 | 0,828 | 0,515 | Ultralytics `val` |
| **Test (1.044 ảnh), đo lại 29/10** | **0,847** | **0,764** | **0,859** | **0,537** | Ultralytics `val` |
| Validation, epoch cuối (55) | 0,840 | 0,723 | 0,808 | 0,467 | `results.csv` |

Nhận xét:
- Model phát hiện người khá tốt (mAP50 > 0,8). Test cao hơn validation một chút, không có dấu hiệu quá khớp vào tập validation.
- Recall 0,72–0,76: khoảng 1/4 người vẫn bị bỏ sót, thường là người xa, bị che hoặc ngược sáng. Cần theo dõi khi thử trên robot thật.
- Từ epoch 35, mAP50-95 giảm nhẹ trong khi loss train vẫn giảm; Ultralytics tự lưu `best.pt` theo epoch tốt nhất nên model đang dùng là epoch 35.
- Trên 314 video cử chỉ của dự án, YOLO tìm thấy người trong mọi frame (tỉ lệ frame có người = 1,00).
- Tốc độ ~4,7 ms/ảnh ở 640 trên GPU RTX 3050 Ti; trên CPU robot chậm hơn nhiều.
- Model trong ROS: `models/best.pt` (5,3 MB, giống hệt `robot_person_detect/weights/best.pt`), ngưỡng `conf = 0,4`.

---

## 4. Mô hình 2: MediaPipe Pose – lấy khung xương

- Dùng bản `pose_landmarker_lite.task` có sẵn của Google, **không huấn luyện thêm**.
- Chạy trên ảnh đã cắt quanh người, trả về 33 điểm `(x, y, visibility)`; sau đó chọn **15 khớp** theo bố cục JHMDB và chuẩn hoá (gốc là trung điểm hai vai, đơn vị là độ dài thân).
- Cùng bản model (`lite`) phải dùng cho cả lúc trích dữ liệu train và lúc chạy trên robot; tên model được ghi vào checkpoint để phát hiện lệch.

**Chất lượng đo gián tiếp** (YOLO + MediaPipe trên 314 video, lần trích `keypoints_v3`):

| Nhóm | Số video | Frame có pose dùng được (trung bình) | Thấp nhất | Số video đạt 100% |
|---|---|---|---|---|
| Robot | 230 | 0,999 | 0,967 | 221/230 |
| RoCoG | 84 | 1,000 | 0,988 | 83/84 |
| Tất cả | 314 | 1,000 | 0,967 | 304/314 |

Không video nào dưới ngưỡng loại 0,5. Kết quả này đạt được sau khi sửa `person_crop.py`: trước đó cái ghế bị YOLO nhận nhầm là người nên khung xương bám lên ghế ở S11–S13
(68 clip có `ok_ratio` dưới 0,5); chọn người theo `conf × độ gần giữa khung` đã đưa `ok_ratio` trung bình của 68 clip này từ 0,23 lên 0,99.

---

## 5. Mô hình 3: DD-Net – nhận diện cử chỉ

### 5.1. Vai trò
Nhận chuỗi 32 khung xương (3,2 giây ở 10 FPS) và phân loại thành 1 trong 5 cử chỉ.

### 5.2. Kiến trúc
DD-Net (Double-feature Double-motion Network, Yang et al. 2019), viết lại bằng PyTorch từ bản Keras gốc. Bản "lite" với 16 filters:
- 3 nhánh đặc trưng: khoảng cách giữa mọi cặp khớp (JCD), chuyển động chậm, chuyển động nhanh; mỗi nhánh dùng Conv1D.
- Ghép nhánh → 3 khối Conv1D → Global Max Pool → 2 lớp fully connected (128) → 5 lớp đầu ra.
- Đặc trưng JCD và chuyển động được tính **bên trong model**, nên lúc train, lúc chạy robot và khi xuất ONNX không thể lệch cách tính.
- Trọng số ~0,62 MB, 145.701 tham số.

### 5.3. Dữ liệu
Hai dataset ghép lại, chia theo **người** để tránh rò rỉ giữa các tập (chi tiết, ID người và số video từng lớp: [DATASET_ROBOT.md](DATASET_ROBOT.md)):

| Tập | Video robot | Video RoCoG | Tổng | Người robot | Người RoCoG |
|---|---|---|---|---|---|
| Train | 112 | 46 | **158** | S11, S12, S13, S14 | S01, S03, S05, S07, S08 |
| Validation | 41 | 12 | **53** | S15 | S02, S10 |
| Test | 77 | 28 | **105** | S16, S17 | S00, S04 (gồm S04b), S06, S09 |
| **Tổng** | 230 | 86 | **316** | 7 người | 11 người |

| Lớp | Train | Validation | Test |
|---|---|---|---|
| MoveForward | 17 | 10 | 18 |
| Halt | 55 | 17 | 35 |
| MoveInReverse | 43 | 12 | 26 |
| Left | 20 | 7 | 15 |
| Right | 23 | 7 | 11 |

- Video robot: 2 m, 5 s, 30 FPS, 640x480. Left, Right, MoveForward chỉ có từ data robot (RoCoG chỉ còn Halt và MoveInReverse).
- MoveForward của RoCoG đã bị gỡ khỏi các file chia ngày 28/10, và S13, S14 không có clip MoveForward, nên MoveForward trong train chỉ đến từ S11 và S12.
- Mỗi video được chuyển thành khung xương rồi lưu `.npz`; resample về 10 FPS, cắt cửa sổ 32 frame (bỏ 0,3 giây đầu/cuối vì nhãn nhiễu).
- Bộ video model dùng, xếp đúng cấu trúc code đọc: `Human Detection/Dataset/Dataset_Used/`.

### 5.4. Cấu hình huấn luyện (`v3_5cls`)
| Tham số | Giá trị |
|---|---|
| Chế độ | Train từ đầu (scratch), không pretrain |
| Filters / Dropout / FC dropout | 16 / 0,1 / 0,5 |
| Batch / lr / weight decay | 64 / 1e-3 / 1e-4 |
| Epoch tối đa / patience | 300 / 40 (chạy 45 epoch, tốt nhất ở epoch 5, ~14 giây trên GPU) |
| Mất mát | Cross-entropy, label smoothing 0,1, class weight |
| Cửa sổ lấy mẫu | 8 cửa sổ ngẫu nhiên mỗi clip mỗi epoch |
| Tăng cường dữ liệu | Xoay ±10°, co giãn 0,9–1,1, đổi tốc độ 0,8–1,2, nhiễu Gauss, che ngẫu nhiên 1 khớp; **tắt** lật gương (lật sẽ lẫn Left với Right) |
| Thiết bị / seed | CUDA / 42 |

### 5.5. Kết quả

| Tập | Số video | Độ chính xác theo video | Macro-F1 | Độ chính xác theo cửa sổ | Loss |
|---|---|---|---|---|---|
| Validation | 53 | 98,1% (52/53) | 0,976 | 98,3% (236 cửa sổ) | 0,137 |
| **Test** | 105 | **100%** (105/105) | **1,000** | 98,7% (455 cửa sổ) | 0,148 |

- **Test:** mọi lớp đạt precision, recall, F1 bằng 1,00; cả 6 người (S00, S04, S06, S09, S16, S17) đều đúng 100%.
- **Validation:** lỗi duy nhất là 1 video MoveForward của S15 bị đoán là Right.

**So với các lần train trước:**

| Lần train | Lớp | Dữ liệu test | Độ chính xác test (video) | Macro-F1 |
|---|---|---|---|---|
| `scratch_f16` | 7 lớp RoCoG | 100 video RoCoG | 95,0% | 0,950 |
| `v2_5cls` | 5 lớp | 117 video (có 14 MoveForward của RoCoG) | 99,1% | 0,993 |
| **`v3_5cls`** | 5 lớp | 105 video | **100%** | **1,000** |

Ba lần dùng ba tập test khác nhau (khác lớp, khác dữ liệu), nên **không thể kết luận model tốt dần lên**. Chưa có lần chạy nào đánh giá `v2` và `v3` trên cùng một tập test.
Ở `scratch_f16`, các lỗi chính là `Rally` ↔ `Attention` (hai cử chỉ vẫy tay giống nhau) và `MoveInReverse` ↔ `MoveForward`; hai lớp Rally, Attention không còn trong bộ lớp hiện tại.

---

## 6. Thành phần ByteTrack và khoá "người chủ"

- ByteTrack là thuật toán ghép detection giữa các frame (Kalman + IoU), **không có trọng số nên không huấn luyện**. Phần có trọng số là YOLO đã finetune ở mục 3.
- Mỗi người có ID ổn định. Máy trạng thái `IDLE → SEARCHING → LOCKED → LOST` có thể khoá một "người chủ" bằng một cử chỉ rồi chỉ nghe lệnh của người này.
  **Hiện đang tắt** (`enable_target_lock: false`, ai cũng điều khiển được), vì cử chỉ khoá `FollowMe` đã bị bỏ khỏi bộ lớp; mặc định `lock_gesture` nay là `Right`.
- Dự đoán quỹ đạo từng người bằng bộ lọc Kalman vận tốc không đổi (`trajectory_node`), chuẩn bị cho phần tránh người sau này.

---

## 7. Kiểm thử tích hợp

Các kiểm thử dưới đây làm với **model 7 lớp cũ** (trước 23/10). Chưa chạy lại với `v3_5cls`.

| Nội dung kiểm thử | Kết quả |
|---|---|
| `colcon build` cả workspace (venv, ROS 2 Jazzy) | Thành công |
| Video `Halt` qua ROS | Ra lệnh **Halt** đúng 100% frame |
| Video `MoveForward` qua ROS | Ra lệnh **MoveForward** đúng 100% frame |
| Video `MoveInReverse` qua ROS | Ra **MoveInReverse** (229 lần), nhầm sang MoveForward 79 lần |
| Tốc độ xử lý trên CPU | ~10 FPS |
| Tracking: nhánh cử chỉ có/không có tracker | Kết quả giống hệt (150/150 frame) |
| Khoá người chủ trên video ghép 2 người | Khoá đúng người làm cử chỉ khoá và giữ khoá |

Cử chỉ cần ~3,2 giây để ra kết quả đầu tiên: đây là đặc điểm thiết kế (cửa sổ 32 frame), không phải lỗi.

---

## 8. Thay đổi từ 23/10 đến 29/10

1. **23/10:** quay thêm video bằng camera robot (230 clip, 7 người); bỏ lớp `None` và `FollowMe`, thêm `Left`, `Right` (5 lớp); sửa lỗi chọn người trong `person_crop.py`; chia theo người; train `v2_5cls` (test 99,1% trên 117 video).
2. **28/10:** thay bộ MoveForward robot bằng 45 clip mới, gỡ MoveForward của RoCoG khỏi các file chia (lý do chưa được ghi lại); trích lại keypoint (`keypoints_v3`); train `v3_5cls` và chép vào `models/ddnet_gesture.pt`. Thêm ngưỡng hiển thị `None` khi xác suất thấp (`--conf`, mặc định 0,8).
3. **29/10:** xếp lại video model dùng vào `Dataset/Dataset_Used/` (316 video) và kiểm tra bằng chính code đọc dữ liệu; viết lại `DATASET_ROBOT.md`; đo lại YOLO trên tập test; thống kê chất lượng MediaPipe; viết `KET_QUA_HUAN_LUYEN.md`.

---

## 9. Công việc tiếp theo

1. Triển khai lên robot (workspace riêng `~/gesture_ws`), đo FPS và CPU của cả pipeline với model `v3` khi chạy chung với SLAM/Navigation.
2. Kiểm thử với người thật: một người, nhiều người, người lạ, phòng khác, khoảng cách khác 2 m, và lúc không làm cử chỉ; chỉnh `conf_threshold`/`confirm_count`.
3. Quay thêm MoveForward của S13, S14 (hoặc người mới), train lại, và đánh giá `v2` với `v3` trên cùng một tập test.
4. Nếu YOLO bỏ sót người trên robot thật, thu thêm ảnh từ camera robot để finetune bổ sung.
5. Làm phần bám người và tránh người dựa trên quỹ đạo đã dự đoán.

---

## 10. Kết luận

Đã hoàn thành chuỗi mô hình nhận diện cử chỉ chạy được trên CPU: **YOLOv5n** tìm người (mAP50 0,86 trên test), **MediaPipe Pose** lấy khung xương (99,96% frame dùng được),
**DD-Net lite** phân loại 5 cử chỉ (test 105 video: 100%, Macro-F1 1,000; validation 98,1%) và **ByteTrack** gán ID.
Kết quả trên dữ liệu thu sẵn rất tốt, nhưng chưa được kiểm chứng với người và môi trường mới; bước tiếp theo quan trọng nhất là thử trên robot thật.

**Các tệp liên quan**
- YOLO: `Human Detection/runs/detect/runs/detect/robot_person_detect/` (`results.csv`, `args.yaml`, `weights/best.pt`)
- DD-Net: `Human Detection/Human Detection/checkpoints/v3_5cls/` (`args.json`, `log.csv`, `results_val.json`, `results_all_test.json`, `predictions_all_test.csv`, `confusion_all_test.png`)
- Dữ liệu: `Human Detection/Dataset/Dataset_Used/` (316 video model dùng), `Human Detection/Dataset/real_ground/` (bản đầy đủ)
- Keypoint: `Human Detection/Human Detection/data/keypoints_v3/` (kèm `extract_report.csv`)
- Mã huấn luyện: `finetune_human.py`, `extract_keypoints.py`, `train_ddnet.py`, `evaluate_ddnet.py`, `ddnet_model.py`
- Package ROS: `ros2_amr_UI-main/src/human_detection/`
- Tài liệu: `KET_QUA_HUAN_LUYEN.md`, `DATASET_ROBOT.md`, `NHAT_KY.md`
