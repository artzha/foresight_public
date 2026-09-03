import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Planner-only launch.

    The vLLM server is launched separately (vllm_server.launch.py for the
    cotrain pipeline, ecot_deployment.launch.py for ECoT) so that the model
    choice and the planner config can vary independently. Point
    `planner_config` at the YAML matching the active server:
      - waypoint_planner.yaml      cotrain motion + critic + reflection
      - ecot_waypoint_planner.yaml ECoT thinking + motion (no critic)
    """
    share_dir = get_package_share_directory("legged_deployment")
    planner_config_default = os.path.join(share_dir, "config", "waypoint_planner.yaml")

    planner_config = LaunchConfiguration("planner_config")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "planner_config",
                default_value=planner_config_default,
                description=(
                    "Path to the waypoint_planner ROS2 parameters YAML. "
                    "Defaults to waypoint_planner.yaml (cotrain pipeline); "
                    "pass ecot_waypoint_planner.yaml for ECoT thinking mode."
                ),
            ),
            Node(
                package="legged_deployment",
                executable="waypoint_planner",
                name="cotnav_waypoint_planner",
                output="screen",
                parameters=[planner_config],
            ),
        ]
    )
