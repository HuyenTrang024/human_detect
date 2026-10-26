
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Thư mục chứa file model .task (cùng thư mục với yolo26n.pt)
WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"

# model_complexity -> tên bản model của MediaPipe (giữ nguyên ý nghĩa cũ: 0 = Lite nhẹ nhất)
MODEL_NAMES = {0: "lite", 1: "full", 2: "heavy"}

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_{name}/float16/latest/pose_landmarker_{name}.task"
)


class PoseEstimator:
    def __init__(self, model_complexity=0, min_det_conf=0.5, min_track_conf=0.5,
                 model_path=None):
        # model_complexity=0 -> bản "Lite", nhẹ nhất, phù hợp real-time trên CPU
        # (mediapipe >= 0.10.30 đã bỏ mp.solutions, nên dùng PoseLandmarker của mp.tasks)
        name = MODEL_NAMES[model_complexity]
        if model_path is None:
            model_path = WEIGHTS_DIR / f"pose_landmarker_{name}.task"
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy model pose: {model_path}\n"
                f"Tải về bằng: curl -L -o \"{model_path}\" {MODEL_URL.format(name=name)}"
            )

        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            # IMAGE: mỗi lần gọi là một ảnh độc lập, không giữ trạng thái tracking.
            # Hợp với ảnh crop vì vị trí crop thay đổi từng frame và không cần reset
            # giữa các video.
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=min_det_conf,
            min_pose_presence_confidence=min_det_conf,
            min_tracking_confidence=min_track_conf,
        )
        self.pose = vision.PoseLandmarker.create_from_options(options)

    def get_keypoints(self, crop_bgr):
        """
        crop_bgr: ảnh crop 1 người (BGR).
        return: list 33 tuple (x, y, visibility) theo pixel của ảnh crop,
                hoặc None nếu không phát hiện được pose.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return None

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.pose.detect(mp_image)
        if not result.pose_landmarks:
            return None

        h, w, _ = crop_bgr.shape
        kps = [
            (lm.x * w, lm.y * h, lm.visibility)
            for lm in result.pose_landmarks[0]
        ]
        return kps

    def close(self):
        self.pose.close()
