# legged_deployment

ROS2 `ament_python` package for service-based language waypoint planning. See the
[repository README](../README.md) for installation, weights, and the launch
workflow.

## `waypoint_planner`

- subscribes to a compressed image topic and an odometry topic,
- buffers a pose-spaced window of the last `obs_window_size` observations,
- serves `amrl_msgs/srv/ForesightPlannerSrv`,
- runs the motion / critic / refinement loop against a vLLM server,
- grounds the predicted image-space trajectory into metric waypoints,
- returns a `nav_msgs/Path` in `path_frame_id`, stamped with the observation image's
  timestamp.

### Interface

| Direction | Name | Type |
| --- | --- | --- |
| Service | `/legged_deployment/foresight_planner` | `amrl_msgs/srv/ForesightPlannerSrv` |
| Subscriber | `image_topic` | `sensor_msgs/CompressedImage` |
| Subscriber | `pose_topic` | `nav_msgs/Odometry` |
| Publisher | `foresight_status_topic` | `amrl_msgs/ForesightPlannerMsg` |
| Publisher | `image_plan_topic` | `sensor_msgs/CompressedImage` |
| Publisher | `obs_mosaic_topic` | `sensor_msgs/CompressedImage` |

`foresight_status_topic` is published once per refinement round with that round's
`reflection_id`, `verdict`, and `reason`, so consumers can follow the self-critique
loop live. `image_plan_topic` carries the annotated trajectory next to a BEV plot of
the grounded waypoints; `obs_mosaic_topic` carries the 2x2 observation window
(oldest to newest, row-major) exactly as the model saw it.

### Reflection loop

Each service call runs up to `max_reflections + 1` rounds. Round 0 uses the `motion`
and `critic` prompts; later rounds use `motion_refine` and `critic_refine`. A round
ends the loop when the critic returns `verdict=1`. With no critic prompt configured
the verdict defaults to true, so the loop returns the first motion proposal.

## LLM backends

Inference is configured through the `llm_cfg.*` ROS parameters.

- `llm_cfg.backend: "vllm_remote"` (default) posts to a running vLLM server at
  `llm_cfg.host:llm_cfg.port` via `/generate`.
- `llm_cfg.backend: "trt_llm"` is scaffolding for TensorRT-LLM. It requires
  `llm_cfg.runtime_callable` to name a Python callable that executes TensorRT
  inference, and raises if the TensorRT runtime is unavailable.

## Build

From the workspace root:

```bash
colcon build --packages-select legged_deployment
source install/setup.bash
```

## Run

Via launch (recommended, see the repository README for variant selection):

```bash
ros2 launch legged_deployment deployment.launch.py
```

Or directly:

```bash
ros2 run legged_deployment waypoint_planner --ros-args \
  --params-file src/foresight_public/legged_deployment/config/waypoint_planner.yaml
```

Request a plan:

```bash
ros2 service call /legged_deployment/foresight_planner \
  amrl_msgs/srv/ForesightPlannerSrv "{goal_command: {data: 'go to the next doorway'}}"
```
