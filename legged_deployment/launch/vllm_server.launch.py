import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import OpaqueFunction
from launch.actions import ExecuteProcess
from launch.substitutions import LaunchConfiguration
from legged_deployment.vllm_config_loader import load_vllm_hydra_overrides


def _launch_vllm_server(context):
    vllm_repo_dir = LaunchConfiguration("vllm_repo_dir").perform(context)
    vllm_server_script = LaunchConfiguration("vllm_server_script").perform(context)
    vllm_config = LaunchConfiguration("vllm_config").perform(context)
    overrides = load_vllm_hydra_overrides(vllm_config)
    return [
        ExecuteProcess(
            cmd=["python3", vllm_server_script, *overrides],
            cwd=vllm_repo_dir,
            output="screen",
        )
    ]


def generate_launch_description() -> LaunchDescription:
    """Launch the vLLM model server that backs the waypoint planner.

    `vllm_config` selects the model variant:
      - vllm_server_sft.yaml supervised fine-tuned weights (default)
      - vllm_server_rl.yaml  RL post-trained weights
    """
    share_dir = get_package_share_directory("legged_deployment")
    default_vllm_repo = os.path.expanduser("~/foresight_ws/src/foresight_public")
    default_vllm_script = os.path.join(
        default_vllm_repo, "scripts", "interactive", "launch_vllm_server.py"
    )
    vllm_config_default = os.path.join(share_dir, "config", "vllm_server_sft.yaml")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "vllm_repo_dir",
                default_value=default_vllm_repo,
                description="foresight_public repository root, used as the working directory for vLLM server startup.",
            ),
            DeclareLaunchArgument(
                "vllm_server_script",
                default_value=default_vllm_script,
                description="Path to the Python script that starts the vLLM server.",
            ),
            DeclareLaunchArgument(
                "vllm_config",
                default_value=vllm_config_default,
                description=(
                    "Path to vLLM server YAML config file. Defaults to "
                    "vllm_server_sft.yaml; pass vllm_server_rl.yaml for the RL "
                    "post-trained variant."
                ),
            ),
            OpaqueFunction(function=_launch_vllm_server),
        ]
    )
