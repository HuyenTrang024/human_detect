#!/usr/bin/env bash
# Cai ROS 2 Jazzy + phu thuoc cho 2 package camera va human_detection (Ubuntu 24.04)
set -euo pipefail

echo "=== 1. Locale + universe repo ==="
sudo apt-get update
sudo apt-get install -y locales software-properties-common curl
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
sudo add-apt-repository -y universe

echo "=== 2. Them ROS 2 apt source ==="
if ! dpkg -s ros2-apt-source >/dev/null 2>&1; then
  ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')
  DEB=$(mktemp --suffix=.deb)
  curl -L -o "$DEB" "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo "$VERSION_CODENAME")_all.deb"
  sudo dpkg -i "$DEB"
  rm -f "$DEB"
fi
sudo apt-get update

echo "=== 3. Cai ROS 2 Jazzy + cac goi can thiet ==="
sudo apt-get install -y \
  ros-jazzy-ros-base \
  ros-dev-tools \
  ros-jazzy-vision-msgs \
  ros-jazzy-rqt-image-view \
  ros-jazzy-librealsense2 \
  libopencv-dev python3-opencv python3-numpy \
  python3-venv

echo "=== 4. Thu vien Python cho gesture: KHONG cai o day ==="
# torch / ultralytics / mediapipe duoc cai trong venv rieng (khong cai vao Python he thong):
#   ros2_amr_UI-main/src/human_detection/setup_venv.sh

echo "=== 5. Tu dong source ROS trong ~/.bashrc ==="
grep -qxF 'source /opt/ros/jazzy/setup.bash' ~/.bashrc || echo 'source /opt/ros/jazzy/setup.bash' >> ~/.bashrc

echo "=== Kiem tra ==="
source /opt/ros/jazzy/setup.bash
ros2 --help >/dev/null && echo "ros2 OK"
python3 -c "import rclpy, vision_msgs, numpy; print('python OK, numpy', numpy.__version__)"
echo "XONG. Mo terminal moi (hoac: source ~/.bashrc), roi chay setup_venv.sh de cai thu vien gesture."
