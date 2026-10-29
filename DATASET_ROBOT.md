# Dataset huấn luyện cử chỉ: robot (`real/robot`) + RoCoG (`real/ground`)

Model DD-Net 5 lớp (MoveForward, Halt, MoveInReverse, Left, Right) được train trên **hai dataset ghép lại**:
video tự quay bằng camera robot (`record_node.py`) và video RoCoG-v2 (`real_ground`).
Số liệu trong file này đã đối chiếu với dữ liệu thật ngày 28/10 (lần train `v3_5cls`, keypoint `keypoints_v3`),
chỉ tính các video **model thật sự dùng** (bộ sao chép đầy đủ nằm ở `Dataset/Dataset_Used/`).

## Tổng quan hai dataset

| | Robot (`real/robot`) | RoCoG-v2 (`real/ground`) | Tổng |
|---|---|---|---|
| Nguồn | tự quay bằng camera robot | bộ dữ liệu công khai [RoCoG-v2](https://github.com/reddyav1/RoCoG-v2) | |
| Số video dùng | **230** | **86** | **316** |
| Lớp có | MoveForward, Halt, MoveInReverse, Left, Right | chỉ Halt, MoveInReverse | 5 lớp |
| Số người | 7 (S11–S17) | 11 (S00–S10, S04b là lượt quay khác của S04) | 18 |
| Khoảng cách / FPS | 2 m, 5 s, 30 FPS, 640x480 | 10 m, clip ngắn, ảnh vuông 1148x1148 | |
| Video bị bỏ | 2 clip trong `rejected/` | 218/304 video thuộc Advance, Attention, Rally, FollowMe, MoveForward | |

**Số video và người theo từng tập** (train / validation / test chia theo người, không người nào ở hai tập):

| Tập | Video robot | Video RoCoG | **Tổng** | ID người robot (số video) | ID người RoCoG (số video) |
|---|---|---|---|---|---|
| **Train** | 112 | 46 | **158** | S11 (28), S12 (33), S13 (30), S14 (21) | S01 (6), S03 (6), S05 (17), S07 (11), S08 (6) |
| **Validation** | 41 | 12 | **53** | S15 (41) | S02 (6), S10 (6) |
| **Test** | 77 | 28 | **105** | S16 (42), S17 (35) | S00 (6), S04 (6), S04b (4), S06 (6), S09 (6) |
| **Tổng** | **230** | **86** | **316** | 7 người | 11 người |

**Số video mỗi lớp theo từng tập** (robot + RoCoG):

| Lớp | Train | Validation | Test | Tổng | Nguồn |
|---|---|---|---|---|---|
| MoveForward | 17 | 10 | 18 | 45 | chỉ robot |
| Halt | 55 | 17 | 35 | 107 | robot 63 + RoCoG 44 |
| MoveInReverse | 43 | 12 | 26 | 81 | robot 39 + RoCoG 42 |
| Left | 20 | 7 | 15 | 42 | chỉ robot |
| Right | 23 | 7 | 11 | 41 | chỉ robot |
| **Tổng** | **158** | **53** | **105** | **316** | |

Chi tiết từng tập (người nào, bao nhiêu video mỗi lớp) nằm ở mục "Chi tiết từng tập" bên dưới.

---

# Dataset robot (`real/robot`)

## Tóm tắt dataset robot

| Mục | Giá trị |
|---|---|
| Số clip | 230 (`robot_all.txt` khớp 230 file `.mp4` trên đĩa) |
| Người quay | 7 người: S11, S12, S13, S14, S15, S16, S17 |
| Khoảng cách | 2 m (tất cả) |
| Độ dài / FPS | 5 s / 30 FPS, ảnh 640x480 (đo lại bằng OpenCV trên toàn bộ MoveForward và Halt) |
| Số lớp | 5: MoveForward, Halt, MoveInReverse, Left, Right |
| Không có | `None` và `FollowMe` (đã bỏ khỏi bộ lớp của model) |

Mỗi clip có nhịp: đứng yên khoảng 0,4 s, làm cử chỉ khoảng 3,5–4 s, rồi hạ tay.
Nhịp này đo bằng chuyển động cổ tay trên 162 clip tốt từ trước ngày 28/10 (cử chỉ bắt đầu ở 0,4 s median, kết thúc ở 4,0–4,4 s);
chưa đo lại sau khi MoveForward được thay.

| Lớp | Nhãn | Số clip |
|---|---|---|
| MoveForward | 0 | 45 |
| Halt | 1 | 63 |
| MoveInReverse | 2 | 39 |
| Left | 3 | 42 |
| Right | 4 | 41 |

Số clip mỗi người (theo thứ tự MoveForward, Halt, MoveInReverse, Left, Right):

| Người | MF | Halt | MIR | Left | Right | Tổng |
|---|---|---|---|---|---|---|
| S11 | 9 | 6 | 4 | 4 | 5 | 28 |
| S12 | 8 | 8 | 6 | 5 | 6 | 33 |
| S13 | **0** | 10 | 6 | 6 | 8 | 30 |
| S14 | **0** | 7 | 5 | 5 | 4 | 21 |
| S15 | 10 | 11 | 6 | 7 | 7 | 41 |
| S16 | 9 | 13 | 6 | 8 | 6 | 42 |
| S17 | 9 | 8 | 6 | 7 | 5 | 35 |

**S13 và S14 không có clip MoveForward**, nên MoveForward ở tập train chỉ đến từ S11 và S12.

## Hai phòng quay

- **S11–S13:** phòng có ghế và bàn, camera đặt sát sàn, người đứng nhỏ và xa trong khung. Đây là dữ liệu "xa".
- **S14–S17:** phòng sáng hơn, người to hơn trong khung.

Giữ cả hai phòng trong train để model quen với cả người gần và người xa.

## Cấu trúc thư mục

Có hai nơi chứa dữ liệu. Cả hai dùng cùng một kiểu đường dẫn trong file annotation, nên code đọc được cả hai.

**1. Dữ liệu gốc: `Human Detection/Dataset/real_ground/`**

```
real_ground/
├── real/
│   ├── ground/<Lớp>/S01_10m_ground_label4_start3682.mp4   # RoCoG (gồm cả lớp không dùng)
│   └── robot/<Lớp>/S11_2m_Halt_take03.mp4                 # <người>_<khoảng cách>_<lớp>_take<số>
├── annotations/
│   ├── robot_all.txt                                      # 230 dòng: <đường dẫn> <nhãn>
│   ├── robot_train.txt  robot_val.txt  robot_test.txt     # 112 / 41 / 77 dòng
│   ├── all_train.txt  all_val.txt  all_test.txt           # RoCoG + robot, file code dùng để train
│   └── all_*.txt.bak_rocog_mf                             # bản trước khi bỏ MoveForward của RoCoG
├── rejected/                    # clip bị loại (không xoá)
└── old_MoveForward_backup/      # bộ MoveForward robot cũ (45 clip, quay ngày 23/10), đã được thay
```

**2. Bộ video chỉ gồm những gì model dùng: `Human Detection/Dataset/Dataset_Used/`** (bản sao, 316 video, 601 MB)

```
Dataset_Used/
├── annotations/all_train.txt  all_val.txt  all_test.txt   # 158 / 53 / 105 dòng, chỉ gồm 5 lớp
└── real/
    ├── ground/  Halt/ MoveInReverse/                       # RoCoG: chỉ 2 lớp còn dùng
    └── robot/   Halt/ Left/ MoveForward/ MoveInReverse/ Right/
```

Dòng annotation: `real/robot/Left/S11_2m_Left_take01.mp4 3`. Số nhãn theo `CLASS_NAMES` trong `rocog_labels.py`
(khác số nhãn RoCoG gốc; `read_split` tự nhận ra video `real/robot/` để dùng bảng này, và báo lỗi nếu tên thư mục lớp không khớp số nhãn).

## Bảng nhãn

```
0 MoveForward   1 Halt   2 MoveInReverse   3 Left   4 Right
```

Video RoCoG thuộc Advance, Attention, Rally, FollowMe bị bỏ qua khi đọc file chia (không train). Model không có lớp `None`;
lúc chạy robot dùng ngưỡng `conf` trong `command_filter.py` để bỏ dự đoán yếu.

## Làm sạch và thay đổi dữ liệu

1. **Annotation lệch file:** `robot_all.txt` ban đầu có 268 dòng, 38 dòng trỏ tới file đã xoá (17 FollowMe, 20 Right, 1 MoveForward).
   Đã lọc còn khớp file. Bản gốc: `robot_all.txt.bak` (trong `~/robot_gesture_data/annotations/`).
2. **Đánh lại số nhãn** theo bảng 5 lớp ở trên (suy ra từ tên thư mục lớp).
3. **Không cắt clip 5 s.** Cửa sổ train dài 3,2 s và `TRIM_SEC = 0.3` đã bỏ 0,3 s đầu và cuối. Muốn sạch hơn thì đổi `TRIM_SEC` (0,4–0,5), không cắt file.
4. **Lỗi chọn người (S11–S13):** `PersonCropper` chọn bbox to nhất, nên cái ghế bị YOLO nhận nhầm là người thắng người đứng xa,
   và MediaPipe vẽ khung xương lên cái ghế. Có 68 clip bị loại (`ok_ratio` < 0,5). Đã sửa `person_crop.py`: chọn theo
   `conf x độ gần giữa khung`. Chạy lại trên 68 clip: `ok_ratio` trung bình 0,23 → 0,99, 67/68 clip dùng được; clip tốt không bị ảnh hưởng.
   Đừng hạ ngưỡng tin cậy pose để "cứu" clip: khung xương khi đó vẫn bám cái ghế.
5. **Clip bị loại, nằm trong `rejected/`:**
   - `S14_2m_MoveInReverse_take03.mp4` (ngày 23/10): `ok_ratio` 0,027, pose gần như không ra.
   - `S12_2m_MoveForward_take06.mp4` (ngày 28/10): chuyển đi cùng đợt thay MoveForward.
6. **Thay bộ MoveForward robot (28/10):** các file trong `real/robot/MoveForward/` được thay bằng bản mới (nội dung khác bản cũ);
   bản cũ giữ ở `old_MoveForward_backup/`. Bộ mới có 45 clip của S11, S12, S15, S16, S17.
7. **Bỏ MoveForward của RoCoG khỏi các file chia (28/10):** 43 video `real/ground/MoveForward/` (23 train, 6 val, 14 test) bị gỡ khỏi
   `all_train/val/test.txt`; bản cũ nằm ở `all_*.txt.bak_rocog_mf`. Nên MoveForward giờ chỉ có data robot.
   (Lý do không được ghi lại trong repo; mục này chỉ mô tả hiện trạng.)
8. **Kiểm tra sau khi trích keypoint (`keypoints_v3`):** không clip nào có `ok_ratio` dưới 0,5.

## Chi tiết từng tập

Cả ba tập ghép robot với RoCoG thành `all_train.txt`, `all_val.txt`, `all_test.txt`.
Cách chia robot: S11–S14 train, S15 validation, S16–S17 test. RoCoG dùng cách chia chính thức của tác giả
(`real_ground_train_tsonly`, `real_ground_val_tsonly`, `real_ground_test`), đã bỏ lớp không dùng.
Cột lớp viết tắt: MF = MoveForward, MIR = MoveInReverse.

### Tập train: 158 video

| Dataset | Người | MF | Halt | MIR | Left | Right | Tổng |
|---|---|---|---|---|---|---|---|
| Robot | S11 | 9 | 6 | 4 | 4 | 5 | 28 |
| Robot | S12 | 8 | 8 | 6 | 5 | 6 | 33 |
| Robot | S13 | 0 | 10 | 6 | 6 | 8 | 30 |
| Robot | S14 | 0 | 7 | 5 | 5 | 4 | 21 |
| RoCoG | S01 | 0 | 3 | 3 | 0 | 0 | 6 |
| RoCoG | S03 | 0 | 3 | 3 | 0 | 0 | 6 |
| RoCoG | S05 | 0 | 11 | 6 | 0 | 0 | 17 |
| RoCoG | S07 | 0 | 4 | 7 | 0 | 0 | 11 |
| RoCoG | S08 | 0 | 3 | 3 | 0 | 0 | 6 |
| **Tổng** | 9 người | **17** | **55** | **43** | **20** | **23** | **158** |

MoveForward trong train chỉ đến từ S11 và S12 (S13, S14 không có clip MoveForward).
Left và Right chỉ có data robot nên ít; đây là các lớp cần theo dõi.

### Tập validation: 53 video

| Dataset | Người | MF | Halt | MIR | Left | Right | Tổng |
|---|---|---|---|---|---|---|---|
| Robot | S15 | 10 | 11 | 6 | 7 | 7 | 41 |
| RoCoG | S02 | 0 | 3 | 3 | 0 | 0 | 6 |
| RoCoG | S10 | 0 | 3 | 3 | 0 | 0 | 6 |
| **Tổng** | 3 người | **10** | **17** | **12** | **7** | **7** | **53** |

### Tập test: 105 video

| Dataset | Người | MF | Halt | MIR | Left | Right | Tổng |
|---|---|---|---|---|---|---|---|
| Robot | S16 | 9 | 13 | 6 | 8 | 6 | 42 |
| Robot | S17 | 9 | 8 | 6 | 7 | 5 | 35 |
| RoCoG | S00 | 0 | 3 | 3 | 0 | 0 | 6 |
| RoCoG | S04 | 0 | 3 | 3 | 0 | 0 | 6 |
| RoCoG | S04b | 0 | 2 | 2 | 0 | 0 | 4 |
| RoCoG | S06 | 0 | 3 | 3 | 0 | 0 | 6 |
| RoCoG | S09 | 0 | 3 | 3 | 0 | 0 | 6 |
| **Tổng** | 7 người | **18** | **35** | **26** | **15** | **11** | **105** |

S04b là lượt quay khác của S04; code gộp chung khi thống kê theo người (S04 = 10 video).

### Dataset RoCoG đã lọc từ bộ gốc như thế nào

Bộ chia gốc có 304 video (162 train + 42 validation + 100 test) gồm 7 lớp. Model chỉ giữ 2 lớp còn trùng bộ lớp của model:

| Tập | Gốc | Giữ lại (Halt + MIR) | Bỏ |
|---|---|---|---|
| Train | 162 | 46 | 116 (Advance 33, Attention 23, MoveForward 23, FollowMe 21, Rally 16) |
| Validation | 42 | 12 | 30 (mỗi lớp 6) |
| Test | 100 | 28 | 72 (Advance 14, Attention 14, MoveForward 14, FollowMe 15, Rally 15) |

MoveForward của RoCoG được gỡ khỏi file chia ngày 28/10 (mục 7 ở "Làm sạch"), nên MoveForward giờ chỉ có data robot.

## Cách tái tạo

Chạy từ thư mục `Human Detection/`. Mọi lệnh Python chạy dạng module từ `Human Detection/Human Detection/`.

```bash
# 1. Tạo file chia robot theo người
cd "Dataset/real_ground/annotations"
grep -E "/S1[1-4]_" robot_all.txt > robot_train.txt
grep -E "/S15_"     robot_all.txt > robot_val.txt
grep -E "/S1[67]_"  robot_all.txt > robot_test.txt
cat real_ground_train_tsonly.txt robot_train.txt > all_train.txt   # tương tự val, test
# (sau đó gỡ các dòng real/ground/MoveForward khỏi all_*.txt, xem mục 7 ở "Làm sạch")

# 2. Kiểm tra file chia, video và chia theo người (không cần GPU)
cd "../../../Human Detection"
python -m training.rocog_labels --ann-dir ../Dataset/Dataset_Used/annotations \
    --dataset-root ../Dataset/Dataset_Used --splits all_train all_val all_test --check-files
# kết quả mong đợi: 158 / 53 / 105 video, tìm thấy 316/316, không trùng video hay người

# 3. Trích keypoint, train, đánh giá (mặc định của --kp-dir/--out-dir trong config.py là data/keypoints_v3; ghi rõ ở đây cho dễ đọc)
python -m training.extract_keypoints --splits all_train all_val all_test --device 0 \
    --ann-dir ../Dataset/Dataset_Used/annotations --dataset-root ../Dataset/Dataset_Used --out-dir data/keypoints_v3
python -m training.train_ddnet --mode scratch --train-splits all_train --val-split all_val \
    --ann-dir ../Dataset/Dataset_Used/annotations --kp-dir data/keypoints_v3 --name v3_5cls --device cuda
python -m training.evaluate_ddnet --ckpt checkpoints/v3_5cls/best.pt --split all_test \
    --ann-dir ../Dataset/Dataset_Used/annotations --kp-dir data/keypoints_v3
```
