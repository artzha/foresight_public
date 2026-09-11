# Iterative Reasoning About Clues that Matter for Navigation

[![Project Website](https://img.shields.io/badge/Project-Website-1f6feb?style=flat&logo=githubsponsors&logoColor=white&labelColor=555555)](https://amrl.cs.utexas.edu/foresight/)
[![Read Paper](https://img.shields.io/badge/Read-Paper-b31b1b?style=flat&logo=googledocs&logoColor=white&labelColor=555555)](https://amrl.cs.utexas.edu/foresight/static/pdfs/ForesightPreprint.pdf)

This project implements **Foresight**, a vision-language navigation policy that iteratively discovers instruction-relevant visual clues and refines motion plans for open-world navigation.

![Foresight framework: an observation history and language goal are fed to a motion
planner, whose plan is judged by a motion critic; the accepted plan is lifted to
metric waypoints by a grounding policy.](docs/media/mainfigure.png)

## Overview

Given a natural-language goal and a short history of RGB observations, the model proposes an image-space trajectory, critiques its own proposal, optionally refines it, and grounds the result into metric BEV waypoints for motion control.

This repository currently contains only what is needed to run the released model weights on a robot. Training, data generation, and evaluation code are not included. See [docs/architecture.md](docs/architecture.md) for how the system fits together,
the service interface, the full configuration reference, and the no-refinement andRL variants.

## Setup

ROS2 Humble or newer and a CUDA GPU are assumed. The vLLM server and the grounding policy both need CUDA; a single 2B model at `gpu_memory_utilization: 0.5` fits comfortably on 24 GB.

Clone this repository into a ROS2 workspace as `src/foresight_public`; the config files reference prompt and config paths under that name.

```bash
git clone <this-repo> ~/foresight_ws/src/foresight_public
cd ~/foresight_ws/src/foresight_public
```

### Python environment

Create and activate a virtual environment, then install the package and its dependencies:

```bash
uv venv --python 3.12
source .venv/bin/activate
bash install_all.sh
```

`install_all.sh` installs this repository in editable mode and then its dependencies
(torch, vLLM, `qwen-vl-utils`, hydra, and the `efficientnet_pytorch` /
`depth-anything-v2` encoders used by the grounding policy).

### Build

The planner's service interface lives in a separate interface package. Build [amrl_msgs](https://github.com/ut-amrl/amrl_msgs) into the same workspace (see [docs/architecture.md](docs/architecture.md#service-interface) for the message definitions if you need to vendor your own), then:

```bash
cd ~/foresight_ws
colcon build --packages-select amrl_msgs legged_deployment
source install/setup.bash
```

## Model weights

Two artifacts are needed: the VLM checkpoint and the grounding policy checkpoint.

Download the grounding policy checkpoint into `checkpoints/`:

```bash
mkdir -p checkpoints/waypoint_policy
huggingface-cli download REPLACE_WITH_HF_ORG/foresight-waypoint-policy \
  gtpassthrough_xformer_kp384_48m.ckpt --local-dir checkpoints/waypoint_policy
```

The VLM is loaded by vLLM directly from its Hugging Face ID, so it only needs to be
named in the server config. Replace `REPLACE_WITH_HF_ORG` in
`legged_deployment/config/vllm_server_sft.yaml` with the released org.

## Run

Two terminals, both from the workspace root. The defaults bring up the SFT variant
with self-critique refinement enabled.

```bash
# terminal 1 - model server
ros2 launch legged_deployment vllm_server.launch.py

# terminal 2 - planner
ros2 launch legged_deployment deployment.launch.py
```

Wait for the server to print `Starting vLLM server on http://127.0.0.1:8001`, then
request a plan:

```bash
ros2 service call /legged_deployment/foresight_planner \
  amrl_msgs/srv/ForesightPlannerSrv "{goal_command: {data: 'go to the next doorway'}}"
```

The response carries the planned `nav_msgs/Path` in the `odom` frame along with the critic's verdict and reason.

If the repository is not at `~/foresight_ws/src/foresight_public`, point the server launch at it:

```bash
ros2 launch legged_deployment vllm_server.launch.py \
  vllm_repo_dir:=/path/to/foresight_public
```

## Citation

If you use this code or the model weights in your research, please cite:

```bibtex
@article{zhang2026foresight,
  title={Foresight: Iterative Reasoning About Clues that Matter for Navigation},
  author={Zhang, Arthur and Qi, Carl and Su, Donne and Meng, Xiangyun and Zhang, Amy and Biswas, Joydeep},
  journal={arXiv preprint arXiv:2606.12550},
  year={2026}
}
```

## License

Apache-2.0
