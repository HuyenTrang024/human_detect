# Human Detection / Gesture Recognition (huấn luyện và thử nghiệm)

Chạy mọi lệnh từ thư mục này, dưới dạng module (`python -m ...`).

```
gesture/     Thư viện lõi dùng cả lúc train lẫn lúc chạy trên robot
             config, skeleton_adapter, pose_estimate, person_crop, ddnet_model, gesture_recognizer
training/    Chuẩn bị dữ liệu, train, đánh giá
             rocog_labels, extract_keypoints, gesture_dataset, train_ddnet, evaluate_ddnet,
             finetune_human (YOLO), merge_dataset
apps/        Chạy thử và điều khiển
             camera_source, demo_camera, robot_control, detect_person, test_inference
checkpoints/ Model DD-Net đã train (v3_5cls là bản 5 lớp đang dùng trên robot)
data/        Keypoint .npz sinh ra khi trích dữ liệu (không đưa lên git)
```

`gesture/` được copy sang gói ROS 2 tại `ros2_amr_UI-main/src/human_detection/human_detection/gesture/`.
Sửa ở đây thì phải đồng bộ sang đó.

## Lệnh thường dùng

```bash
A=../Dataset/Dataset_Used      # 316 video model dùng: train 158, val 53, test 105
python -m training.extract_keypoints --splits all_train all_val all_test --device 0 \
    --ann-dir $A/annotations --dataset-root $A --out-dir data/keypoints_v3
python -m training.train_ddnet --mode scratch --train-splits all_train --val-split all_val \
    --ann-dir $A/annotations --kp-dir data/keypoints_v3 --name v3_5cls --device cuda
python -m training.evaluate_ddnet --ckpt checkpoints/v3_5cls/best.pt --split all_test \
    --ann-dir $A/annotations --kp-dir data/keypoints_v3
python -m apps.demo_camera --ckpt checkpoints/v3_5cls/best.pt
python -m apps.robot_control --ckpt checkpoints/v3_5cls/best.pt --backend print --show
```

`training/merge_dataset.py` và `training/finetune_human.py` là script chạy thẳng khi import,
không dùng như thư viện. Chạy bằng `python -m training.merge_dataset`.
