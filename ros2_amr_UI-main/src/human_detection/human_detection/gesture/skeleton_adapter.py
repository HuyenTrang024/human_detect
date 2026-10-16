"""
skeleton_adapter.py: đổi keypoint MediaPipe thành "khung xương" cho DD-Net.

Đầu vào : list 33 tuple (x, y, visibility) từ PoseEstimator.get_keypoints()
          (toạ độ pixel của ảnh crop, trục y hướng XUỐNG).
Đầu ra  : AdaptedFrame(joints, visible, scale) hoặc None nếu frame không dùng được.
              joints  : mảng (J, 2), toạ độ thân người, đúng thứ tự JOINTS bên dưới
              visible : mảng bool (J,), True = khớp thật sự được MediaPipe nhìn thấy,
                        False = giá trị được điền thay thế
              scale   : số pixel ứng với 1 đơn vị toạ độ (để kiểm tra/debug)

Các việc file này làm:
  1. Chọn khớp và xếp đúng thứ tự mà DD-Net pretrain mong đợi (bảng JOINTS).
  2. Chuẩn hoá toạ độ: gốc = trung điểm hai vai; đơn vị gọi là "độ rộng vai chuẩn".
     Thước đo được lấy từ ĐỘ DÀI THÂN (trung điểm vai -> trung điểm hông) chứ không
     phải độ rộng vai từng frame: khi người xoay nghiêng, độ rộng vai co lại rất
     nhiều còn độ dài thân gần như không đổi. Thước đo còn được làm mượt theo thời
     gian để không rung.
  3. Khớp bị che được điền bằng giá trị của frame trước (hoặc vị trí mặc định).
  4. Cung cấp mirror_joints() để lật ngang dữ liệu (augment) đúng cách:
     đổi dấu x VÀ hoán đổi khớp trái/phải.

Nếu bạn dùng bản DD-Net pretrain có bố cục khớp khác, chỉ cần sửa 3 hằng số:
JOINTS, MIRROR_PAIRS, DEFAULT_POSITIONS.
"""

from typing import NamedTuple, Optional

import numpy as np

# ===========================================================================
# BỐ CỤC KHỚP: sửa ở đây nếu đổi bản DD-Net pretrain
# ===========================================================================
# Chỉ số khớp trong 33 điểm của MediaPipe Pose
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

# Bố cục 15 khớp của bộ dữ liệu JHMDB (bản DD-Net pretrain hay dùng).
# Mỗi khớp = (tên, [các chỉ số MediaPipe để lấy trung bình]).
# Khớp chỉ coi là "thấy" khi TẤT CẢ chỉ số nguồn đều đủ visibility.
#
# LƯU Ý: "right"/"left" ở đây giả định là tay phải/trái CỦA NGƯỜI, giống MediaPipe.
# Hãy kiểm tra lại với nguồn của trọng số pretrain; nếu ngược thì hoán đổi
# L_* và R_* trong bảng này.
JOINTS = [
    ("neck",       [L_SHOULDER, R_SHOULDER]),   # cổ = trung điểm hai vai
    ("belly",      [L_HIP, R_HIP]),             # bụng = trung điểm hai hông
    ("face",       [NOSE]),
    ("r_shoulder", [R_SHOULDER]),
    ("l_shoulder", [L_SHOULDER]),
    ("r_hip",      [R_HIP]),
    ("l_hip",      [L_HIP]),
    ("r_elbow",    [R_ELBOW]),
    ("l_elbow",    [L_ELBOW]),
    ("r_knee",     [R_KNEE]),
    ("l_knee",     [L_KNEE]),
    ("r_wrist",    [R_WRIST]),
    ("l_wrist",    [L_WRIST]),
    ("r_ankle",    [R_ANKLE]),
    ("l_ankle",    [L_ANKLE]),
]

# Các cặp khớp trái/phải, để lật ngang (dùng tên khớp ở bảng JOINTS)
MIRROR_PAIRS = [
    ("r_shoulder", "l_shoulder"),
    ("r_hip", "l_hip"),
    ("r_elbow", "l_elbow"),
    ("r_knee", "l_knee"),
    ("r_wrist", "l_wrist"),
    ("r_ankle", "l_ankle"),
]

# Vị trí mặc định của từng khớp khi một người đứng thẳng, tay buông xuôi.
# Toạ độ thân người (x, y): gốc ở trung điểm hai vai, đơn vị = độ rộng vai,
# y hướng xuống. Chỉ dùng để điền khi khớp bị che mà chưa có dữ liệu trước đó.
# Đây là số ước lượng theo tỉ lệ cơ thể trung bình.
DEFAULT_POSITIONS = {
    "neck":       (0.0, 0.0),
    "belly":      (0.0, 1.3),
    "face":       (0.0, -0.6),
    "r_shoulder": (-0.5, 0.0),
    "l_shoulder": (0.5, 0.0),
    "r_hip":      (-0.4, 1.3),
    "l_hip":      (0.4, 1.3),
    "r_elbow":    (-0.55, 0.8),
    "l_elbow":    (0.55, 0.8),
    "r_knee":     (-0.4, 2.4),
    "l_knee":     (0.4, 2.4),
    "r_wrist":    (-0.55, 1.5),
    "l_wrist":    (0.55, 1.5),
    "r_ankle":    (-0.4, 3.5),
    "l_ankle":    (0.4, 3.5),
}

# ===========================================================================
# CÁC NGƯỠNG
# ===========================================================================
# Khớp có visibility thấp hơn ngưỡng này bị coi là "không thấy"
MIN_VISIBILITY = 0.5

# Độ rộng vai (pixel) nhỏ hơn giá trị này thì không đáng tin.
# Chỉ còn dùng để kiểm tra frame; khi người xoay nghiêng 90 độ thì hai vai gần chồng
# lên nhau, nên ngưỡng này cố ý đặt thấp.
MIN_SHOULDER_WIDTH_PX = 2.0

# Độ dài thân (pixel) từ trung điểm hai vai đến trung điểm hai hông; nhỏ hơn giá
# trị này thì không đáng tin
MIN_TORSO_PX = 10.0

# Tỉ lệ độ dài thân / độ rộng vai của người bình thường. Chia độ dài thân cho số này
# để đổi về cùng đơn vị "độ rộng vai" như trước, nhờ đó DEFAULT_POSITIONS và các
# ngưỡng tính theo độ rộng vai vẫn dùng được. Giá trị này bằng toạ độ y của belly
# trong DEFAULT_POSITIONS. config.py import hằng số này nên chỉ cần sửa ở đây.
TORSO_TO_SHOULDER_RATIO = 1.3

# Làm mượt thước đo theo thời gian (EMA). 1.0 = không làm mượt, nhỏ hơn = mượt hơn.
# Đặt quá nhỏ thì thước đo trễ khi người đi lại gần/ra xa camera.
SCALE_SMOOTHING = 0.3

# Frame bị loại nếu số khớp thật sự nhìn thấy ít hơn số này.
# (Gối và cổ chân thường bị che khi robot chỉ nhìn thân trên, nên không đặt quá cao.)
MIN_VISIBLE_JOINTS = 8

# ===========================================================================
# Giá trị suy ra từ bảng trên
# ===========================================================================
NUM_JOINTS = len(JOINTS)
JOINT_NAMES = [name for name, _ in JOINTS]
_NAME_TO_INDEX = {name: i for i, name in enumerate(JOINT_NAMES)}

# Chỉ số khớp đối xứng: MIRROR_INDEX[i] = chỉ số của khớp bên kia (khớp ở giữa -> chính nó)
MIRROR_INDEX = list(range(NUM_JOINTS))
for _a, _b in MIRROR_PAIRS:
    MIRROR_INDEX[_NAME_TO_INDEX[_a]] = _NAME_TO_INDEX[_b]
    MIRROR_INDEX[_NAME_TO_INDEX[_b]] = _NAME_TO_INDEX[_a]

_DEFAULT_ARRAY = np.array(
    [DEFAULT_POSITIONS[name] for name in JOINT_NAMES], dtype=np.float32
)


class AdaptedFrame(NamedTuple):
    """Kết quả chuyển đổi của một frame."""
    joints: np.ndarray    # (NUM_JOINTS, 2) float32, toạ độ thân người
    visible: np.ndarray   # (NUM_JOINTS,) bool; False = khớp được điền thay thế
    scale: float          # số pixel ứng với 1 đơn vị toạ độ (thước đo đã làm mượt)


def mirror_joints(joints):
    """
    Lật ngang khung xương (dùng cho augment).

    joints: mảng (..., NUM_JOINTS, 2), toạ độ thân người (gốc ở giữa hai vai).
    return: mảng cùng kích thước, đã
        - đổi dấu x (lật qua trục dọc đi qua giữa hai vai),
        - hoán đổi khớp trái <-> phải.
    Khi lật dữ liệu train, nhớ đổi nhãn LEFT <-> RIGHT cho khớp.
    """
    joints = np.asarray(joints)
    mirrored = joints[..., MIRROR_INDEX, :].copy()
    mirrored[..., 0] = -mirrored[..., 0]
    return mirrored


class SkeletonAdapter:
    """
    Đổi keypoint MediaPipe của từng frame thành khung xương cho DD-Net.

    Object này có trạng thái nhỏ: nhớ vị trí gần nhất của từng khớp để điền khi
    khớp bị che, và nhớ thước đo đã làm mượt. Gọi reset() khi mất người, đổi sang
    người khác, hoặc bắt đầu một video mới.

    Dùng đúng cùng một cách này cho cả lúc trích dữ liệu train và lúc chạy trên robot,
    để thước đo giống nhau ở hai nơi.
    """

    def __init__(self, min_visibility=MIN_VISIBILITY,
                 min_visible_joints=MIN_VISIBLE_JOINTS):
        self.min_visibility = min_visibility
        self.min_visible_joints = min_visible_joints
        self._last = _DEFAULT_ARRAY.copy()
        self._scale = None    # thước đo đã làm mượt; None = chưa có

    def reset(self):
        """Quên vị trí và thước đo cũ (gọi khi mất người hoặc bắt đầu video mới)."""
        self._last = _DEFAULT_ARRAY.copy()
        self._scale = None

    def _measure_scale(self, kps, ok):
        """
        Thước đo của frame này (pixel / 1 đơn vị toạ độ), chưa làm mượt.
        Ưu tiên độ dài thân; nếu không thấy hông thì dùng thước đo của frame trước;
        nếu chưa có frame trước thì dùng độ rộng vai.
        """
        if ok[L_HIP] and ok[R_HIP]:
            shoulder_mid = (kps[L_SHOULDER, :2] + kps[R_SHOULDER, :2]) / 2
            hip_mid = (kps[L_HIP, :2] + kps[R_HIP, :2]) / 2
            torso = float(np.hypot(*(hip_mid - shoulder_mid)))
            if torso >= MIN_TORSO_PX:
                return torso / TORSO_TO_SHOULDER_RATIO

        if self._scale is not None:
            return self._scale

        return float(np.hypot(*(kps[L_SHOULDER, :2] - kps[R_SHOULDER, :2])))

    def convert(self, keypoints) -> Optional[AdaptedFrame]:
        """
        keypoints: list 33 tuple (x, y, visibility), hoặc None.
        return: AdaptedFrame, hoặc None nếu frame không dùng được
                (không có keypoint, không thấy hai vai, hoặc quá ít khớp thấy được).
                Khi trả None thì trạng thái nhớ KHÔNG bị thay đổi.
        """
        if keypoints is None or len(keypoints) < 33:
            return None

        kps = np.asarray(keypoints, dtype=np.float32)   # (33, 3): x, y, visibility
        ok = kps[:, 2] >= self.min_visibility

        # --- Bắt buộc phải thấy cả hai vai (làm gốc toạ độ và thước đo) ---
        if not (ok[L_SHOULDER] and ok[R_SHOULDER]):
            return None

        shoulder_width = float(np.hypot(*(kps[L_SHOULDER, :2] - kps[R_SHOULDER, :2])))
        if shoulder_width < MIN_SHOULDER_WIDTH_PX:
            return None

        # --- Thước đo: độ dài thân (làm mượt theo thời gian) ---
        raw_scale = self._measure_scale(kps, ok)
        if self._scale is None:
            scale = raw_scale
        else:
            scale = (1 - SCALE_SMOOTHING) * self._scale + SCALE_SMOOTHING * raw_scale

        # --- Chuẩn hoá: gốc = trung điểm hai vai, đơn vị = "độ rộng vai chuẩn" ---
        origin = (kps[L_SHOULDER, :2] + kps[R_SHOULDER, :2]) / 2
        body = (kps[:, :2] - origin) / scale    # (33, 2)

        # --- Dựng từng khớp của DD-Net từ các điểm nguồn MediaPipe ---
        joints = self._last.copy()    # khớp không thấy giữ nguyên giá trị cũ
        visible = np.zeros(NUM_JOINTS, dtype=bool)
        for i, (_, sources) in enumerate(JOINTS):
            if all(ok[s] for s in sources):
                joints[i] = body[sources].mean(axis=0)
                visible[i] = True

        if visible.sum() < self.min_visible_joints:
            return None

        # Chỉ cập nhật trạng thái khi frame được chấp nhận
        self._last = joints.copy()
        self._scale = scale
        return AdaptedFrame(joints=joints, visible=visible, scale=scale)


# ---------------------------------------------------------------------------
# Chạy thử nhanh: python skeleton_adapter.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    def make_pose(**points):
        """Tạo 33 keypoint giả; khớp không khai báo coi như không thấy."""
        kps = [(0.0, 0.0, 0.0)] * 33
        for index, xy in points.items():
            kps[int(index[1:])] = (xy[0], xy[1], 1.0)
        return kps

    # Ảnh crop: vai trái của người ở x=150, vai phải ở x=50 (độ rộng vai = 100 px).
    # Tay trái giơ cao, tay phải buông. Không thấy gối và cổ chân.
    pose = make_pose(
        k0=(100, 40),
        k11=(150, 100), k12=(50, 100),
        k13=(150, 50), k15=(150, 0),
        k14=(50, 150), k16=(50, 200),
        k23=(130, 250), k24=(70, 250),
    )

    adapter = SkeletonAdapter()
    frame = adapter.convert(pose)
    print("số khớp:", NUM_JOINTS, "| joints.shape:", frame.joints.shape)
    print("khớp được điền thay thế:",
          [JOINT_NAMES[i] for i in range(NUM_JOINTS) if not frame.visible[i]])
    print("l_wrist (tay trái giơ cao, y âm = cao hơn vai):",
          frame.joints[JOINT_NAMES.index("l_wrist")])

    # Lật ngang: tay trái giơ cao phải thành tay phải giơ cao
    flipped = mirror_joints(frame.joints)
    print("sau khi lật, r_wrist:", flipped[JOINT_NAMES.index("r_wrist")])
    print("lật hai lần = ban đầu:", np.allclose(mirror_joints(flipped), frame.joints))

    # Frame không thấy vai -> bị loại
    print("thiếu vai ->", adapter.convert(make_pose(k0=(100, 40))))

    # Người xoay nghiêng: hai vai co từ 100 px xuống 50 px, độ dài thân vẫn như cũ.
    # Thước đo theo độ dài thân phải gần như không đổi (nếu theo độ rộng vai sẽ giảm một nửa).
    front = make_pose(k11=(150, 100), k12=(50, 100), k23=(130, 250), k24=(70, 250),
                      k13=(150, 150), k15=(150, 200), k14=(50, 150), k16=(50, 200))
    side = make_pose(k11=(125, 100), k12=(75, 100), k23=(115, 250), k24=(85, 250),
                     k13=(125, 150), k15=(125, 200), k14=(75, 150), k16=(75, 200))
    scale_front = SkeletonAdapter().convert(front).scale
    scale_side = SkeletonAdapter().convert(side).scale
    print(f"thước đo: nhìn thẳng {scale_front:.1f} px | xoay nghiêng {scale_side:.1f} px")

    # Không thấy hông: dùng thước đo của frame trước thay vì độ rộng vai
    no_hip = make_pose(k11=(125, 100), k12=(75, 100),
                       k13=(125, 150), k15=(125, 200), k14=(75, 150), k16=(75, 200),
                       k0=(100, 40))
    adapter2 = SkeletonAdapter()
    adapter2.convert(front)
    print(f"không thấy hông: thước đo {adapter2.convert(no_hip).scale:.1f} px "
          f"(giữ gần {scale_front:.1f}, không nhảy về 50)")
