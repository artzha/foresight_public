# legged_deployment

ROS2 `ament_python` package for service-based language waypoint planning.

`waypoint_planner`:
- subscribes to image and robot pose topics,
- stores a bounded observation history,
- exposes `LanguagePlannerSrv` (`amrl_msgs/srv/LanguagePlannerSrv`),
- runs language planner + grounding stages,
- returns a `nav_msgs/Path` in `base_link` using the selected observation image timestamp.

LLM inference is configured through `llm_cfg.*` ROS params.

- `llm_cfg.backend: "vllm_remote"` (default) sends requests to a running vLLM
  server (`host`/`port`) via `/generate` using `trajectory:v1`.
- `llm_cfg.backend: "trt_llm"` is scaffolded for future TensorRT-LLM support.
  When enabled, `llm_cfg.runtime_callable` must point to a Python callable that
  executes TensorRT inference for your selected runtime stack.

## Build

From workspace root:

```bash
colcon build --packages-select legged_deployment
source install/setup.bash
```

## Run

Run waypoint planner node:

```bash
ros2 run legged_deployment waypoint_planner --ros-args --params-file src/cotnav/legged_deployment/config/waypoint_planner.yaml
```

Run with launch:

```bash
ros2 launch legged_deployment deployment.launch.py
```

Call service:

```bash
ros2 service call /language_planner amrl_msgs/srv/LanguagePlannerSrv "{goal_command: {data: 'go to the next doorway'}}"
```
