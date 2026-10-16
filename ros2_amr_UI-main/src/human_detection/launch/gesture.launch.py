import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('human_detection')
    config_file = os.path.join(pkg_share, 'config', 'gesture_params.yaml')

    gesture_node = Node(
        package='human_detection',
        executable='gesture_node',
        name='gesture_node',
        output='screen',
        parameters=[config_file]
    )

    return LaunchDescription([
        gesture_node
    ])
