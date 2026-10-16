#!/usr/bin/env bash
# Tao moi truong ao (venv) cho gesture_node: ROS 2 Jazzy, chay CPU.
#
#   ./setup_venv.sh                  # tao o ~/gesture_venv
#   ./setup_venv.sh /duong/dan/venv  # tao o cho khac
#
# Sau do moi lan chay:
#   source /opt/ros/jazzy/setup.bash
#   source ~/gesture_venv/bin/activate
#   source <ros2_ws>/install/setup.bash
#   ros2 launch human_detection gesture.launch.py
set -euo pipefail

VENV="${1:-$HOME/gesture_venv}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== 1. Goi he thong (can sudo) ==="
PKGS=()
python3 -c "import ensurepip" 2>/dev/null || PKGS+=(python3-venv)
dpkg -s ros-jazzy-vision-msgs >/dev/null 2>&1 || PKGS+=(ros-jazzy-vision-msgs)
dpkg -s python3-numpy >/dev/null 2>&1 || PKGS+=(python3-numpy)
if [ ${#PKGS[@]} -gt 0 ]; then
  sudo apt-get update
  sudo apt-get install -y "${PKGS[@]}"
else
  echo "Da du."
fi

echo "=== 2. Tao venv: $VENV ==="
# --system-site-packages: van thay thu vien apt ma ROS can (yaml, numpy 1.26, ...)
python3 -m venv --system-site-packages "$VENV"
# Neu venv nam trong ROS workspace thi colcon bo qua no
touch "$VENV/COLCON_IGNORE"

# Khong doc thu vien trong ~/.local (pip --user) de venv khong bi lan phien ban khac
if ! grep -q PYTHONNOUSERSITE "$VENV/bin/activate"; then
  sed -i 's/^deactivate () {$/deactivate () {\n    unset PYTHONNOUSERSITE/' "$VENV/bin/activate"
  echo 'export PYTHONNOUSERSITE=1' >> "$VENV/bin/activate"
fi

echo "=== 3. Cai thu vien Python (torch CPU, ultralytics, mediapipe) ==="
export PYTHONNOUSERSITE=1
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install -r "$HERE/requirements.txt"

echo "=== 4. Kiem tra ==="
source /opt/ros/jazzy/setup.bash
"$VENV/bin/python" - <<'EOF'
import numpy, torch, ultralytics, mediapipe, cv2, rclpy, vision_msgs.msg
assert numpy.__version__.startswith("1."), f"numpy {numpy.__version__} phai < 2"
print(f"OK  numpy {numpy.__version__} | torch {torch.__version__} | ultralytics {ultralytics.__version__} "
      f"| mediapipe {mediapipe.__version__} | cv2 {cv2.__version__} | rclpy OK")
EOF
echo "XONG. Venv: $VENV"
