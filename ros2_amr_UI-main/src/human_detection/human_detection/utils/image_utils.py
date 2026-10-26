"""Chuyển đổi sensor_msgs/Image <-> ảnh numpy BGR, không cần cv_bridge (tránh xung đột numpy)."""

import cv2
import numpy as np
from sensor_msgs.msg import Image


def imgmsg_to_bgr(msg: Image) -> np.ndarray:
    """Chuyen sensor_msgs/Image -> anh BGR (numpy) ma khong can cv_bridge."""
    enc = msg.encoding.lower()
    channels = {'bgr8': 3, 'rgb8': 3, 'bgra8': 4, 'rgba8': 4, 'mono8': 1}.get(enc)
    if channels is None:
        raise ValueError(f'Encoding khong ho tro: {msg.encoding}')

    buf = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    img = buf[:, :msg.width * channels].reshape(msg.height, msg.width, channels)

    if enc == 'rgb8':
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if enc == 'bgra8':
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if enc == 'rgba8':
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    if enc == 'mono8':
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def bgr_to_imgmsg(img: np.ndarray, header) -> Image:
    """Dong goi anh BGR (numpy) -> sensor_msgs/Image (bgr8)."""
    img = np.ascontiguousarray(img)
    msg = Image()
    msg.header = header
    msg.height, msg.width = img.shape[:2]
    msg.encoding = 'bgr8'
    msg.is_bigendian = False
    msg.step = img.shape[1] * 3
    msg.data = img.tobytes()
    return msg
