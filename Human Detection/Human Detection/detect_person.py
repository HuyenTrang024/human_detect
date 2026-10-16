
import config
from person_crop import YoloPersonDetector


class PersonDetector:
    def __init__(
        self,
        weights_path=config.YOLO_WEIGHTS,
        conf=0.5,
        device="cpu",
        person_class_id=0,
    ):
        # Dùng ultralytics (đúng thư viện đã train best.pt) thay cho torch.hub của yolov5,
        # và dùng chung lớp YoloPersonDetector với extract_keypoints.py / PersonCropper
        # để lúc chạy thật cho ra bbox giống lúc trích dữ liệu.
        self.detector = YoloPersonDetector(
            weights_path, device=device, conf=conf, person_class_id=person_class_id
        )

    def detect(self, frame_bgr):
        """
        frame_bgr: ảnh BGR (OpenCV) từ camera, 1 frame.
        return: list các bbox (x1, y1, x2, y2, conf), đã sort giảm dần theo
                diện tích bbox (người ở gần nhất / to nhất lên đầu).
        """
        return [
            (int(x1), int(y1), int(x2), int(y2), float(conf))
            for x1, y1, x2, y2, conf in self.detector.detect(frame_bgr)
        ]
