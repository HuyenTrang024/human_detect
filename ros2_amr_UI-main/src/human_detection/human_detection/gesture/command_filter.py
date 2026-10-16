"""
command_filter.py - Chọn lệnh ổn định từ chuỗi dự đoán của DD-Net (lấy từ robot_control.py).

An toàn:
  - Lệnh chỉ đổi khi cùng một cử chỉ có xác suất >= conf trong `confirm` lần dự đoán liên tiếp.
  - Mất người / mất khung xương quá lost_sec -> về "Attention" (đứng yên).
  - Không có cử chỉ mới thì giữ lệnh tối đa hold_sec rồi về "Attention".
  - Halt luôn được chấp nhận ngay (confirm = 1 cho Halt).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Cử chỉ -> chuyển động. SỬA BẢNG NÀY theo ý nghĩa bạn muốn.  (vx m/s, wz rad/s)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Motion:
    vx: float = 0.0
    wz: float = 0.0
    follow: bool = False       # FollowMe: bật chế độ bám theo người (chưa làm)


GESTURE_TO_MOTION: Dict[str, Motion] = {
    "MoveForward":   Motion(vx=0.20),
    "Advance":       Motion(vx=0.20),
    "MoveInReverse": Motion(vx=-0.15),
    "Halt":          Motion(),
    "Attention":     Motion(),          # đứng yên, chờ lệnh
    "Rally":         Motion(),          # ví dụ: về điểm tập kết -> tự gắn hành vi riêng
    "FollowMe":      Motion(follow=True),
}
STOP = Motion()


class CommandFilter:
    def __init__(self, conf: float, confirm: int, hold_sec: float, lost_sec: float):
        self.conf, self.confirm, self.hold_sec, self.lost_sec = conf, confirm, hold_sec, lost_sec
        self.current = "Attention"
        self._cand, self._count = None, 0
        self._last_set = time.time()
        self._last_seen = time.time()

    def step(self, label: Optional[str], prob: float, now: float) -> str:
        if label is None:                                     # chưa có dự đoán / mất người
            self._cand, self._count = None, 0
            if now - self._last_seen > self.lost_sec:
                self.current = "Attention"
            return self.current
        self._last_seen = now

        if prob >= self.conf and label != self.current:
            self._count = self._count + 1 if label == self._cand else 1
            self._cand = label
            need = 1 if label == "Halt" else self.confirm
            if self._count >= need:
                self.current, self._last_set = label, now
                self._cand, self._count = None, 0
        elif label == self.current and prob >= self.conf:
            self._last_set = now                              # vẫn đang làm cử chỉ đó -> giữ lệnh
            self._cand, self._count = None, 0
        else:
            self._cand, self._count = None, 0

        if now - self._last_set > self.hold_sec:              # lâu không thấy cử chỉ -> dừng
            self.current = "Attention"
        return self.current
