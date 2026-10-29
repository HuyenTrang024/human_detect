# Kết quả các mô hình nhận diện người và cử chỉ

Pipeline của robot gồm bốn khối: **YOLO** tìm người, **ByteTrack** gán ID, **MediaPipe Pose** lấy khung xương, **DD-Net** phân loại cử chỉ.
File này mô tả đầu vào/đầu ra của từng khối và kết quả đạt được. Dữ liệu và cách chia tập: xem [DATASET_ROBOT.md](DATASET_ROBOT.md).

Nguồn số liệu: `runs/detect/runs/detect/robot_person_detect/results.csv` (YOLO train), `Human Detection/checkpoints/v3_5cls/` (DD-Net),
`data/keypoints_v3/extract_report.csv` (YOLO + MediaPipe trên 316 video), và hai phép đo tôi chạy lại ngày 29/10 (đánh dấu "đo lại").

---

## 1. Cấu trúc tổng quan

```
Ảnh camera (BGR) ─► YOLO ─► ByteTrack ─► Cắt ảnh người ─► MediaPipe Pose ─► Chuẩn hoá khung xương
 /image_front     bbox người   ID người    crop quanh bbox    33 điểm (x, y, vis)    15 khớp (x, y)

   ─► Cửa sổ 32 frame @10 FPS (3,2 s) ─► DD-Net ─► xác suất 5 lớp ─► CommandFilter ─► /gesture
          (32, 15, 2)                    (B, 5)    làm mượt EMA 0,5     ngưỡng conf
```

| Khối | Đầu vào | Đầu ra | Có huấn luyện? | Kích thước |
|---|---|---|---|---|
| **YOLO** (YOLOv5n finetune, 1 lớp `Person`) | 1 ảnh BGR cả khung hình, resize `imgsz` 640 | danh sách bbox người `(x1, y1, x2, y2, conf)`, giữ `conf` ≥ 0,4 | Có (finetune 55 epoch) | 5,3 MB |
| **ByteTrack** | các bbox của YOLO theo từng frame | bbox kèm ID người ổn định giữa các frame | Không (thuật toán, không có trọng số) | – |
| **Cắt ảnh người** (`PersonCropper`) | ảnh gốc + bbox người được chọn | ảnh crop quanh người (nới bbox 25% mỗi phía, làm mượt bbox EMA 0,5, giữ bbox cũ tối đa 5 frame nếu YOLO trượt) | Không | – |
| **MediaPipe Pose** (`pose_landmarker_lite`) | ảnh crop 1 người (RGB) | 33 điểm `(x, y, visibility)`, toạ độ pixel của ảnh crop; `None` nếu không thấy pose | Không (model có sẵn của Google) | 5,8 MB |
| **Chuẩn hoá khung xương** (`SkeletonAdapter`) | 33 điểm của MediaPipe | 15 khớp `(x, y)` theo bố cục JHMDB; gốc toạ độ là trung điểm hai vai, đơn vị là độ dài thân; khớp bị che được điền bằng giá trị frame trước | Không | – |
| **DD-Net** (lite, 16 filters) | tensor `(B, 32, 15, 2)`: 32 frame × 15 khớp × (x, y), resample về 10 FPS | logits `(B, 5)`, qua softmax thành xác suất 5 lớp: MoveForward, Halt, MoveInReverse, Left, Right | Có (train từ đầu) | 0,62 MB, 145.701 tham số |

Ghi chú về từng khối:

- **DD-Net** tự tính đặc trưng khoảng cách giữa mọi cặp khớp (JCD) và hai nhánh chuyển động (chậm, nhanh) ngay trong model, nên lúc train và lúc chạy robot không thể lệch cách tính.
  Cấu trúc: 3 nhánh Conv1D → ghép → 3 khối Conv1D → Global Max Pool → 2 lớp fully connected 128 → 5 lớp đầu ra.
- Một cử chỉ cần ~3,2 s dữ liệu mới ra kết quả đầu tiên; đây là thiết kế của cửa sổ 32 frame, không phải lỗi.
- Trên robot, `GestureRecognizer` làm mượt xác suất theo thời gian bằng EMA 0,5, bỏ cửa sổ nếu dưới 60% frame có khung xương dùng được.
- ByteTrack chỉ nằm trong nhánh ROS (`people_tracking.launch.py`); lúc huấn luyện DD-Net không dùng.

---

## 2. Kết quả YOLO (phát hiện người)

**Dữ liệu:** `Merged_Human_Dataset`, 9.375 ảnh, 1 lớp `Person`, gộp từ 3 bộ (Human Dataset 2.188 ảnh, person_1 1.484 ảnh, person_2 5.703 ảnh).

| Tập | Số ảnh |
|---|---|
| Train | 6.288 |
| Validation | 2.043 |
| Test | 1.044 |

**Cấu hình:** khởi tạo từ `yolov5n.pt` (COCO), ảnh 640, batch 16, optimizer auto, lr0 0,01, tối đa 60 epoch / patience 20.
Chạy 55 epoch (~32 phút), `best.pt` lấy ở epoch 35.

**Kết quả của `best.pt`:**

| Tập | Precision | Recall | mAP50 | mAP50-95 | Nguồn |
|---|---|---|---|---|---|
| Validation, epoch 35 | 0,856 | 0,721 | 0,826 | 0,514 | `results.csv` khi train |
| Validation, đo lại | 0,856 | 0,721 | 0,828 | 0,515 | Ultralytics `val`, `conf=0,001` |
| **Test (1.044 ảnh), đo lại** | **0,847** | **0,764** | **0,859** | **0,537** | Ultralytics `val`, `conf=0,001` |
| Validation, epoch cuối (55) | 0,840 | 0,723 | 0,808 | 0,467 | `results.csv` |

- Test cao hơn validation một chút (mAP50 0,859 so với 0,828), nên không có dấu hiệu quá khớp vào tập validation. Từ epoch 35 mAP50-95 giảm nhẹ trong khi loss train vẫn giảm, nên `best.pt` ở epoch 35 là lựa chọn đúng.
- Tốc độ suy luận đo lại: ~4,7 ms/ảnh ở 640 trên GPU RTX 3050 Ti (robot chạy CPU, chậm hơn nhiều).
- Recall 0,72–0,76: khoảng 1/4 người vẫn bị bỏ sót, thường là người xa, bị che hoặc ngược sáng.
- **Trên 316 video cử chỉ của dự án**, YOLO tìm thấy người trong mọi frame của cả 314 video đã trích (`person_ratio` = 1,00 ở tất cả).
  Con số này đo trên người đứng 2 m (robot) hoặc 10 m (RoCoG) chiếm khá lớn trong khung, nên dễ hơn tập ảnh tổng quát ở trên.
- Model trong ROS (`models/best.pt`) giống hệt `robot_person_detect/weights/best.pt`.

---

## 3. Kết quả MediaPipe Pose (khung xương)

Dùng nguyên model `pose_landmarker_lite` của Google, **không huấn luyện thêm**, nên không có độ chính xác riêng.
Chất lượng được đo gián tiếp bằng tỉ lệ frame mà YOLO + MediaPipe cho ra khung xương dùng được (`extract_report.csv`, lần trích `keypoints_v3`):

| Nhóm | Số video | Tỉ lệ frame có người | Tỉ lệ frame có pose (trung bình) | Thấp nhất | Số video đạt 100% |
|---|---|---|---|---|---|
| Robot | 230 | 1,000 | 0,999 | 0,967 | 221/230 |
| RoCoG | 84 | 1,000 | 1,000 | 0,988 | 83/84 |
| **Tất cả** | **314** | **1,000** | **1,000** | **0,967** | **304/314** |

- Không video nào dưới ngưỡng loại 0,5 (`MIN_OK_RATIO`); video xấu nhất vẫn có 96,7% frame dùng được.
- 2 video RoCoG (MoveInReverse của S01) có sẵn từ lần trích trước nên không có thống kê trong lần chạy `v3`; chúng vẫn nằm trong train.
- Phần này đạt được **sau khi sửa `person_crop.py`**: trước đó, ghế bị YOLO nhận nhầm là người nên MediaPipe vẽ khung xương lên ghế ở S11–S13,
  68 clip có `ok_ratio` dưới 0,5. Sau sửa, `ok_ratio` trung bình của 68 clip này tăng từ 0,23 lên 0,99.
- Thời gian trích keypoint trung bình ~5,4 giây mỗi video (cột `seconds` trong `extract_report.csv`; không ghi lại máy dùng).

---

## 4. Kết quả DD-Net (nhận diện cử chỉ, `v3_5cls`)

### Tóm tắt

| Tập | Số video | Độ chính xác theo video | Macro-F1 | Độ chính xác theo cửa sổ | Loss |
|---|---|---|---|---|---|
| Validation (S02, S10, S15) | 53 | **98,1%** (52/53) | 0,976 | 98,3% (236 cửa sổ) | 0,137 |
| **Test** (S00, S04, S06, S09, S16, S17) | 105 | **100%** (105/105) | **1,000** | 98,7% (455 cửa sổ) | 0,148 |

"Theo video" là kết quả sau khi gộp các cửa sổ của một video; "theo cửa sổ" tính từng cửa sổ 3,2 s (bước trượt 4 frame).
Test gồm 6 người chưa từng xuất hiện trong train; chia tập theo người, không có người hay video nào nằm ở hai tập.

Tốc độ (đo lại): một cửa sổ `(1, 32, 15, 2)` chạy hết ~1,4 ms trên 1 luồng CPU, nên DD-Net không phải điểm nghẽn; phần nặng là YOLO và MediaPipe.

### Cấu hình huấn luyện

| Mục | Giá trị |
|---|---|
| Chế độ | Train từ đầu (`scratch`), không pretrain |
| Dữ liệu train | `all_train`: 158 video (robot 112 + RoCoG 46) |
| Keypoint | `data/keypoints_v3`, `pose_landmarker_lite` |
| Model | DD-Net lite, 16 filters, dropout 0,1, FC dropout 0,5 |
| Cửa sổ | 32 frame @ 10 FPS = 3,2 s, bỏ 0,3 s đầu và cuối clip |
| Tối ưu | batch 64, lr 1e-3, weight decay 1e-4, label smoothing 0,1, class weight |
| Tăng cường | xoay ±10°, co giãn 0,9–1,1, đổi tốc độ 0,8–1,2, nhiễu Gauss, che 1 khớp; tắt lật gương |
| Dừng | early stopping patience 40: chạy 45 epoch, tốt nhất ở **epoch 5**, tổng ~14 giây trên GPU |
| Chọn checkpoint | `best.pt` theo tập validation |
| Seed | 42 |

### Kết quả theo lớp

**Test (105 video):** không nhầm video nào.

| Lớp | Precision | Recall | F1 | Số video |
|---|---|---|---|---|
| MoveForward | 1,00 | 1,00 | 1,00 | 18 |
| Halt | 1,00 | 1,00 | 1,00 | 35 |
| MoveInReverse | 1,00 | 1,00 | 1,00 | 26 |
| Left | 1,00 | 1,00 | 1,00 | 15 |
| Right | 1,00 | 1,00 | 1,00 | 11 |

Theo người trên test: cả 6 người đều 100% (số video: S00 6, S04 10 gồm cả S04b, S06 6, S09 6, S16 42, S17 35).

**Validation (53 video):** có 1 lỗi duy nhất.

| Lớp | Precision | Recall | F1 | Số video |
|---|---|---|---|---|
| MoveForward | 1,00 | 0,90 | 0,947 | 10 |
| Halt | 1,00 | 1,00 | 1,00 | 17 |
| MoveInReverse | 1,00 | 1,00 | 1,00 | 12 |
| Left | 1,00 | 1,00 | 1,00 | 7 |
| Right | 0,875 | 1,00 | 0,933 | 7 |

Lỗi: 1 video MoveForward của S15 bị đoán là Right. Theo người: S02 100%, S10 100%, S15 97,6% (40/41).

### So với các lần train trước

| Lần train | Lớp | Dữ liệu test | Độ chính xác test (video) | Macro-F1 |
|---|---|---|---|---|
| `scratch_f16` | 7 lớp RoCoG | 100 video RoCoG | 95,0% | 0,950 |
| `v2_5cls` | 5 lớp | 117 video (có 14 MoveForward của RoCoG) | 99,1% | 0,993 |
| **`v3_5cls`** | 5 lớp | 105 video | **100%** | **1,000** |

**Không nên đọc bảng này như "model tốt dần lên".** Tập test của ba lần không giống nhau: `v2_5cls` còn video MoveForward của RoCoG
(lỗi duy nhất của nó là 1 video loại này của S00 bị đoán thành MoveInReverse), còn `v3_5cls` đã bỏ các video đó và dùng bộ MoveForward robot mới.
Chưa có lần chạy nào đánh giá `v2` và `v3` trên cùng một tập test.

---

## 5. Cách đọc kết quả cho đúng

- **100% trên test DD-Net không có nghĩa là 100% ngoài thực tế.** Test chỉ có 105 video, trong đó 77 là của S16, S17 quay cùng thiết bị, cùng khoảng cách 2 m và
  gần giống phòng với S14, S15 trong train/val. Bài toán cũng chỉ còn 5 lớp khác nhau rõ về động tác.
- **Các số liệu hai khối đầu (YOLO, MediaPipe) tốt hơn trên video cử chỉ** vì người đứng rõ, một người trong khung; robot thật có thể gặp nhiều người, người ở xa hơn, bị che.
- **Model chưa có lớp `None`.** Khi không ai làm cử chỉ, model vẫn phải chọn 1 trong 5 lớp; robot lọc bằng ngưỡng `conf` trong `command_filter.py`. Các con số trên chưa đo phần này.
- **MoveForward có ít người trong train:** chỉ S11 và S12 (17 clip); S13, S14 không có clip MoveForward, và các video MoveForward của RoCoG đã bị bỏ.
  Lớp này duy nhất có lỗi ở validation.
- **Left, Right** chỉ có từ data robot (20 và 23 clip train), nhưng đều đúng 100% trên test.
- **Early stopping ở epoch 5** với tập validation chỉ 53 video: lựa chọn checkpoint có thể chưa ổn định.
- **Chưa đo lại toàn pipeline với model `v3`.** Kết quả chạy end-to-end trong ROS (~10 FPS trên CPU, Halt và MoveForward đúng 100% frame trên video)
  nằm trong `BAO_CAO_HUAN_LUYEN.md` và đo với model 7 lớp cũ.

## 6. Việc nên làm tiếp để biết chất lượng thật

1. Thử trên robot với người ngoài S11–S17, phòng khác, khoảng cách khác 2 m, nhiều người trong khung và lúc không làm cử chỉ.
2. Đo lại tốc độ (FPS) và CPU của cả pipeline với model `v3`, khi chạy chung với SLAM/Navigation.
3. Quay thêm MoveForward của S13, S14 (hoặc người mới) rồi train lại; đánh giá `v2` và `v3` trên cùng một tập test để so sánh công bằng.
4. Nếu YOLO bỏ sót người trên robot thật (recall chỉ ~0,72–0,76), thu thêm ảnh từ camera robot để finetune bổ sung.

## 7. Tái hiện kết quả

Chạy DD-Net từ `Human Detection/Human Detection/`:

```bash
python -m training.evaluate_ddnet --ckpt checkpoints/v3_5cls/best.pt --split all_test \
    --ann-dir ../Dataset/Dataset_Used/annotations --kp-dir data/keypoints_v3
```

Kết quả mong đợi: 105 video, độ chính xác 100%. Từ 29/10, `KEYPOINT_DIR` mặc định trong `config.py` đã là `data/keypoints_v3`; `--kp-dir` ghi ở trên để rõ ràng.

Đánh giá YOLO trên tập test (cần môi trường có `ultralytics`, ví dụ conda `human_detect`). `data.yaml` gốc trỏ tới đường dẫn của máy khác
(`/media/itvkist/...`), nên cần một file yaml có `path` đúng ở máy này:

```bash
yolo val model="Human Detection/runs/detect/runs/detect/robot_person_detect/weights/best.pt" \
    data=<yaml có path trỏ tới Dataset/Merged_Human_Dataset> split=test imgsz=640 conf=0.001
```

Kết quả mong đợi: precision 0,847, recall 0,764, mAP50 0,859, mAP50-95 0,537.
