from __future__ import annotations

import io
from pathlib import Path as FsPath
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import rclpy
from amrl_msgs.msg import ForesightPlannerMsg
from amrl_msgs.srv import ForesightPlannerSrv
from nav_msgs.msg import Odometry, Path
from PIL import Image as PILImage
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Header, String
from tf2_ros import Buffer, TransformException, TransformListener

from foresight.prompts.interface import ChatQuery, OutputFormat
from foresight.prompts.prompt_utils import (
    build_critic_segment,
    build_motion_segment,
    build_thinking_segment,
)
import foresight.utils.hydra_utils as hu
from foresight.utils.draw import draw_polyline, draw_bev_poses_topdown
from foresight.geometry.camera import crop_np, plan_resize_center_crop
from legged_deployment.language_planner_model import (
    CritiqueOutput,
    LanguagePlannerModel,
    PlannerLLMConfig,
    PlannerOutput,
)
from legged_deployment.observation_history import ObservationHistory
from legged_deployment.path_utils import (
    se3_waypoints_to_path,
    transform_se3_waypoints,
    xyz_waypoints_to_se3,
)


def _image_to_numpy(msg: CompressedImage) -> np.ndarray:
    if not msg.data:
        raise ValueError("Compressed image payload is empty.")
    pil = PILImage.open(io.BytesIO(bytes(msg.data))).convert("RGB")
    return np.asarray(pil, dtype=np.uint8)


def _parse_image_resolution(value: object) -> Optional[Tuple[int, int]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if len(value) != 2:
        return None
    try:
        height = int(value[0])
        width = int(value[1])
    except (TypeError, ValueError):
        return None
    if height <= 0 or width <= 0:
        return None
    return (height, width)


def _resolve_prompt_text(raw_value: object) -> str:
    value = str(raw_value or "").strip()
    if not value:
        return ""
    path = FsPath(value).expanduser()
    candidates = [path, FsPath.cwd() / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8").strip()
    return value


def _resize_center_crop_bottom(image_np: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    in_h, in_w = int(image_np.shape[0]), int(image_np.shape[1])
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    plan = plan_resize_center_crop(
        (in_h, in_w),
        (target_h, target_w),
        vertical_anchor="bottom",
        bottom_margin_px=0.0,
    )
    pre_h, pre_w = plan["pre_hw"]
    resized = np.asarray(
        PILImage.fromarray(image_np).resize((pre_w, pre_h), resample=PILImage.BILINEAR),
        dtype=np.uint8,
    )
    return crop_np(
        resized[np.newaxis, ...],
        (target_h, target_w),
        plan["crop_offset_uv"],
    )[0]


def _to_compressed_image_msg(image_np: np.ndarray, header: object) -> CompressedImage:
    pil = PILImage.fromarray(image_np.astype(np.uint8), mode="RGB")
    buffer = io.BytesIO()
    pil.save(buffer, format="JPEG", quality=90)
    msg = CompressedImage()
    msg.header = header
    msg.format = "jpeg"
    msg.data = buffer.getvalue()
    return msg


def _build_obs_mosaic_2x2(images: Sequence[np.ndarray]) -> np.ndarray:
    """Tile up to four observations into a 2x2 grid (oldest -> newest, row-major).

    Slots are filled by repeating the latest image when fewer than four frames
    are available. All tiles are resized to the latest tile's shape so the
    grid is rectangular regardless of source heterogeneity.
    """
    if not images:
        raise ValueError("Cannot build mosaic from empty images list.")
    latest = images[-1]
    target_hw = (int(latest.shape[0]), int(latest.shape[1]))
    tiles: List[np.ndarray] = list(images)[-4:]
    while len(tiles) < 4:
        tiles.insert(0, latest)

    def _match_shape(tile: np.ndarray) -> np.ndarray:
        if tile.shape[:2] == target_hw:
            return tile
        resized = PILImage.fromarray(tile.astype(np.uint8)).resize(
            (target_hw[1], target_hw[0]), resample=PILImage.BILINEAR
        )
        return np.asarray(resized, dtype=np.uint8)

    resized_tiles = [_match_shape(t) for t in tiles]
    top = np.hstack([resized_tiles[0], resized_tiles[1]])
    bottom = np.hstack([resized_tiles[2], resized_tiles[3]])
    return np.vstack([top, bottom])


class WaypointPlannerNode(Node):
    def __init__(self) -> None:
        super().__init__("waypoint_planner")

        self.declare_parameter("image_topic", "/camera/rgb/image_raw/compressed")
        self.declare_parameter("pose_topic", "/laser_odometry")

        self.declare_parameter("image_plan_topic", "/legged_deployment/image_plan")
        self.declare_parameter(
            "obs_mosaic_topic", "/legged_deployment/observation_mosaic/compressed"
        )
        self.declare_parameter(
            "foresight_status_topic", "/legged_deployment/foresight_status"
        )
        self.declare_parameter(
            "service_name", "/legged_deployment/foresight_planner"
        )
        self.declare_parameter("path_frame_id", "base_link")
        self.declare_parameter("obs_frame_id", "base_link")
        self.declare_parameter("image_resolution", [224, 392])
        self.declare_parameter("max_reflections", 1)
        self.declare_parameter("obs_window_size", 4)
        self.declare_parameter("obs_window_spacing_m", 1.06)
        self.declare_parameter("prompts.system", "")
        self.declare_parameter("prompts.motion", "")
        self.declare_parameter("prompts.critic", "")
        self.declare_parameter("prompts.motion_refine", "")
        self.declare_parameter("prompts.thinking", "")
        self.declare_parameter("llm_cfg.backend", "vllm_remote")
        self.declare_parameter("llm_cfg.max_new_tokens", 256)
        self.declare_parameter("llm_cfg.host", "127.0.0.1")
        self.declare_parameter("llm_cfg.port", 8001)
        self.declare_parameter("llm_cfg.timeout_s", 180.0)
        self.declare_parameter("llm_cfg.auth_token", "")
        self.declare_parameter("llm_cfg.engine_path", "")
        self.declare_parameter("llm_cfg.tokenizer_name", "")
        self.declare_parameter("llm_cfg.runtime_callable", "")
        self.declare_parameter("llm_cfg.debug_io", False)
        self.declare_parameter("grounding_policy.enabled", True)
        self.declare_parameter("grounding_policy.config_dir", "src/foresight/configs")
        self.declare_parameter(
            "grounding_policy.config_name", "model/waypoint/gtpassthrough_simple"
        )
        self.declare_parameter("grounding_policy.checkpoint_path", "")
        self.declare_parameter("grounding_policy.device", "cuda")
        self.declare_parameter("grounding_policy.strict_load", False)
        self.declare_parameter("grounding_policy.action_head.num_kp", 0)
        self.declare_parameter("grounding_policy.action_head.dim_feedforward", 0)
        self.declare_parameter("grounding_policy.action_head.nhead", 0)
        self.declare_parameter("grounding_policy.action_head.num_layers", 0)

        self._path_frame_id = str(self.get_parameter("path_frame_id").value)
        self._obs_frame_id = str(
            self.get_parameter("obs_frame_id").value
        )
        self._image_resolution = _parse_image_resolution(
            self.get_parameter("image_resolution").value
        )
        self._max_reflections = max(0, int(self.get_parameter("max_reflections").value))
        self._obs_window_size = max(1, int(self.get_parameter("obs_window_size").value))
        self._obs_window_spacing_m = float(
            self.get_parameter("obs_window_spacing_m").value
        )
        # Deque holds (window_size - 1) pose-spaced past snapshots; the live
        # latest is appended as the final element at inference time.
        self._history = ObservationHistory(max_len=max(0, self._obs_window_size - 1))
        self._latest_pose_msg: Optional[Odometry] = None
        self._tf_buffer = Buffer(cache_time=Duration(seconds=60.0))
        self._tf_listener = TransformListener(self._tf_buffer, self)

        raw_prompt_params = self.get_parameters_by_prefix("prompts")
        self.prompts: Dict[str, str] = {}
        for key, param in raw_prompt_params.items():
            prompt_text = _resolve_prompt_text(param.value)
            key_name = str(key).strip().lower()
            if prompt_text:
                self.prompts[key_name] = prompt_text
                self.get_logger().info(f"Loaded prompt '{key_name}'.")

        llm_cfg = PlannerLLMConfig(
            backend=str(self.get_parameter("llm_cfg.backend").value),
            max_new_tokens=int(self.get_parameter("llm_cfg.max_new_tokens").value),
            host=str(self.get_parameter("llm_cfg.host").value),
            port=int(self.get_parameter("llm_cfg.port").value),
            timeout_s=float(self.get_parameter("llm_cfg.timeout_s").value),
            auth_token=str(self.get_parameter("llm_cfg.auth_token").value),
            engine_path=str(self.get_parameter("llm_cfg.engine_path").value),
            tokenizer_name=str(self.get_parameter("llm_cfg.tokenizer_name").value),
            runtime_callable=str(self.get_parameter("llm_cfg.runtime_callable").value),
        )

        grounding_policy_requested = bool(
            self.get_parameter("grounding_policy.enabled").value
        )
        grounding_cfg = None
        if grounding_policy_requested:
            grounding_cfg = self._configure_grounding_policy()
            assert grounding_cfg is not None, "Failed to configure grounding policy."

        self._planner_model = LanguagePlannerModel(
            llm_cfg=llm_cfg,
            grounding_policy_cfg=grounding_cfg,
        )

        image_plan_topic = str(self.get_parameter("image_plan_topic").value)
        obs_mosaic_topic = str(self.get_parameter("obs_mosaic_topic").value)
        foresight_status_topic = str(
            self.get_parameter("foresight_status_topic").value
        )
        service_name = str(self.get_parameter("service_name").value)
        self._image_plan_publisher = self.create_publisher(CompressedImage, image_plan_topic, 10)
        self._obs_mosaic_publisher = self.create_publisher(
            CompressedImage, obs_mosaic_topic, 10
        )
        self._foresight_status_publisher = self.create_publisher(
            ForesightPlannerMsg, foresight_status_topic, 10
        )

        image_topic = str(self.get_parameter("image_topic").value)
        pose_topic = str(self.get_parameter("pose_topic").value)
        self.create_subscription(CompressedImage, image_topic, self._on_image, 10)
        self.create_subscription(Odometry, pose_topic, self._on_pose, 30)
        self.create_service(ForesightPlannerSrv, service_name, self._on_plan)

        self.get_logger().info(f"Publishing image plan to topic: {image_plan_topic}")
        self.get_logger().info(
            f"Publishing observation mosaic to topic: {obs_mosaic_topic}"
        )
        self.get_logger().info(
            f"Publishing foresight status to topic: {foresight_status_topic}"
        )
        self.get_logger().info(
            f"Waypoint source frame: {self._obs_frame_id}; path frame: {self._path_frame_id}"
        )
        self.get_logger().info(f"Listening image topic: {image_topic}")
        self.get_logger().info(f"Listening pose topic: {pose_topic}")
        if self._image_resolution is not None:
            h, w = self._image_resolution
            self.get_logger().info(f"Resizing planner images to {h}x{w} (height x width).")
        self.get_logger().info(
            f"Loaded prompt templates: {sorted(self.prompts.keys())}"
        )
        self.get_logger().info(f"Max reflection iterations: {self._max_reflections}")
        if grounding_cfg is not None:
            self.get_logger().info("Grounding policy configured in LanguagePlannerModel.")
        self.get_logger().info(f"Service ready: {service_name}")

    def _configure_grounding_policy(self) -> None:
        config_dir = str(self.get_parameter("grounding_policy.config_dir").value).strip()
        config_name = str(self.get_parameter("grounding_policy.config_name").value).strip()
        checkpoint_path = str(
            self.get_parameter("grounding_policy.checkpoint_path").value
        ).strip()
        strict_load = bool(self.get_parameter("grounding_policy.strict_load").value)
        device = str(self.get_parameter("grounding_policy.device").value).strip()
        try:
            cfg = hu.hydra_compose(
                config_path=config_dir,
                config_name=config_name,
            )
            raw_cfg = dict(hu.resolve(cfg))
            model_cfg = dict(raw_cfg["model"]["waypoint"])

            # Override action_head transformer kwargs from ROS params (0 = keep default).
            num_kp = int(self.get_parameter("grounding_policy.action_head.num_kp").value)
            ff_dim = int(self.get_parameter("grounding_policy.action_head.dim_feedforward").value)
            nhead = int(self.get_parameter("grounding_policy.action_head.nhead").value)
            num_layers = int(self.get_parameter("grounding_policy.action_head.num_layers").value)
            action_head = dict(model_cfg.get("action_head") or {})
            ah_kwargs = dict(action_head.get("kwargs") or {})
            if num_kp > 0:
                ah_kwargs["num_kp"] = num_kp
                # spatial_softmax.num_kp must match action_head.num_kp.
                spatial_encoder = dict(model_cfg.get("spatial_encoder") or {})
                spatial_softmax = dict(spatial_encoder.get("spatial_softmax") or {})
                spatial_softmax["num_kp"] = num_kp
                spatial_encoder["spatial_softmax"] = spatial_softmax
                model_cfg["spatial_encoder"] = spatial_encoder
            if ff_dim > 0:
                ah_kwargs["dim_feedforward"] = ff_dim
            if nhead > 0:
                ah_kwargs["nhead"] = nhead
            if num_layers > 0:
                ah_kwargs["num_layers"] = num_layers
            action_head["kwargs"] = ah_kwargs
            model_cfg["action_head"] = action_head

            enc_input_res = (
                model_cfg.get("spatial_encoder", {})
                .get("encoder", {})
                .get("input_res")
            )
            rgb_resolution = None
            if isinstance(enc_input_res, (list, tuple)) and len(enc_input_res) == 2:
                rgb_resolution = [int(enc_input_res[0]) * 14, int(enc_input_res[1]) * 14]
            grounding_cfg = {
                "model_cfg": model_cfg,
                "checkpoint_path": checkpoint_path,
                "strict_load": strict_load,
                "device": device,
                "rgb_resolution": rgb_resolution,
            }
        except Exception as exc:
            self.get_logger().error(
                "Failed to compose grounding config "
                f"(config_dir='{config_dir}', config_name='{config_name}'): {exc}"
            )
        return grounding_cfg

    def _on_pose(self, msg: Odometry) -> None:
        self._latest_pose_msg = msg

    def _on_image(self, msg: CompressedImage) -> None:
        if self._latest_pose_msg is None:
            self.get_logger().debug("Image skipped: waiting for first pose message.")
            return
        try:
            image_np = _image_to_numpy(msg)
        except Exception as exc:
            self.get_logger().warn(f"Failed to decode image: {exc}")
            return
        self._history.update_pose(
            self._latest_pose_msg,
            msg,
            image_np,
            self._obs_window_spacing_m,
        )

    @staticmethod
    def _pose_to_xyz(pose: Odometry) -> Tuple[float, float, float]:
        p = pose.pose.pose.position
        return float(p.x), float(p.y), float(p.z)

    def _transform_xyz_waypoints(
        self,
        xyz_waypoints: Sequence[Sequence[float]],
        *,
        pose_msg,
    ) -> Sequence[Tuple[float, float, float, float]]:
        """Build SE3 (x, y, z, yaw) waypoints in the path frame.

        Thin wrapper over :func:`legged_deployment.path_utils`: derive a
        successor-pointing yaw per waypoint (drops the last), then rotate the
        body-frame poses into ``self._path_frame_id`` using the pose captured
        atomically with the image. See :func:`xyz_waypoints_to_se3` and
        :func:`transform_se3_waypoints` for the math.
        """
        local_se3 = xyz_waypoints_to_se3(xyz_waypoints)
        if not local_se3:
            return []
        return transform_se3_waypoints(
            local_se3,
            pose_msg=pose_msg,
            obs_frame_id=self._obs_frame_id,
            path_frame_id=self._path_frame_id,
        )

    def _publish_iteration_artifacts(
        self,
        *,
        base_image_np: np.ndarray,
        pixel_waypoints: List[List[float]],
        local_xyz_waypoints: np.ndarray,
        header,
        path_msg: Path,
        reflection_id: int,
        verdict: bool,
        reason: str,
        thinking_text: str = "",
        motion_text: str = "",
        critic_text: str = "",
    ) -> None:
        annotated = draw_polyline(
            pixel_waypoints,
            base_image_np.copy(),
            color=(51, 255, 255),
            line_thickness=2,
            dot_radius=3,
        )
        if local_xyz_waypoints.size > 0:
            H, W = annotated.shape[:2]
            bev_image = draw_bev_poses_topdown(
                local_xyz_waypoints,
                colors=[(51, 255, 255)],
                image_hw=(H, W),
                xlim=(-6.4, 6.4),
                ylim=(-6.4, 6.4),
                title=f"Predicted BEV Actions (refl {reflection_id})",
            )
            annotated = np.hstack([annotated, bev_image])
        motion_image_msg = _to_compressed_image_msg(annotated, header)
        # Keep the image_plan publisher for backwards compatibility with
        # downstream consumers (e.g. monitor) that subscribe to the topic.
        self._image_plan_publisher.publish(motion_image_msg)

        status_msg = ForesightPlannerMsg()
        status_msg.header = Header()
        status_msg.header.stamp = header.stamp
        status_msg.header.frame_id = self._path_frame_id
        status_msg.path = path_msg
        status_msg.reflection_id = int(reflection_id)
        status_msg.verdict = Bool(data=bool(verdict))
        status_msg.reason = String(data=str(reason))
        status_msg.thinking_text = String(data=str(thinking_text))
        status_msg.motion_text = String(data=str(motion_text))
        status_msg.critic_text = String(data=str(critic_text))
        # Embed a copy of the annotated motion image so chat-style consumers
        # can render it next to the assistant's motion message without having
        # to subscribe to the (separately published) image_plan topic.
        status_msg.motion_image = motion_image_msg
        self._foresight_status_publisher.publish(status_msg)

    def _waypoints_to_path_msg(
        self,
        xyz_waypoints: Sequence[Sequence[float]],
        *,
        stamp,
        pose_msg,
    ) -> Tuple[Path, np.ndarray]:
        local_xyz = np.asarray(xyz_waypoints, dtype=float)
        if local_xyz.size == 0:
            return Path(), local_xyz
        before_lines = "\n".join(
            f"  [{i}] x={p[0]:.3f} y={p[1]:.3f} z={p[2]:.3f}"
            for i, p in enumerate(local_xyz)
        )
        self.get_logger().info(
            f"Path before transform ({self._obs_frame_id}, n={len(local_xyz)}):\n{before_lines}"
        )
        se3_transformed = self._transform_xyz_waypoints(local_xyz, pose_msg=pose_msg)
        after_lines = "\n".join(
            f"  [{i}] x={p[0]:.3f} y={p[1]:.3f} z={p[2]:.3f} yaw={p[3]:.3f}"
            for i, p in enumerate(se3_transformed)
        )
        self.get_logger().info(
            f"Path after transform ({self._path_frame_id}, n={len(se3_transformed)}):\n{after_lines}"
        )
        path_msg = se3_waypoints_to_path(
            se3_transformed, stamp=stamp, frame_id=self._path_frame_id
        )
        return path_msg, local_xyz

    def _on_plan(
        self,
        request: ForesightPlannerSrv.Request,
        response: ForesightPlannerSrv.Response,
    ) -> ForesightPlannerSrv.Response:
        obs = self._history.latest()
        response.foresight_plan = ForesightPlannerMsg()
        response.foresight_plan.header.frame_id = self._path_frame_id
        response.foresight_plan.path = Path()
        response.foresight_plan.reflection_id = 0
        response.foresight_plan.verdict = Bool(data=False)
        response.foresight_plan.reason = String(data="")

        if obs is None:
            response.foresight_plan.reason = String(
                data="No observation history available."
            )
            self.get_logger().warn(
                "No observation history available for planning request."
            )
            return response

        goal_text = str(request.goal_command.data)

        try:
            window_snaps = self._history.inference_window(self._obs_window_size)
            assert window_snaps, "Empty observation window."

            def _prepare_image(image_np: np.ndarray) -> np.ndarray:
                if self._image_resolution is None:
                    return image_np
                target_h, target_w = self._image_resolution
                return _resize_center_crop_bottom(image_np, (target_h, target_w))

            window_images = [_prepare_image(snap.image_np) for snap in window_snaps]
            base_image_np = window_images[-1]
            pil_frames = [
                PILImage.fromarray(img).convert("RGB") for img in window_images
            ]
            stamp = obs.image_msg.header.stamp
            header = obs.image_msg.header

            # Publish the 2x2 observation mosaic that mirrors the planner's
            # input window (oldest -> newest, row-major). Consumers (e.g.
            # webviz) display this as the right-panel image so operators can
            # see exactly what the model just saw.
            try:
                mosaic_np = _build_obs_mosaic_2x2(window_images)
                self._obs_mosaic_publisher.publish(
                    _to_compressed_image_msg(mosaic_np, header)
                )
            except Exception as exc:
                self.get_logger().warn(f"Failed to publish observation mosaic: {exc}")

            system_prompt = self.prompts.get("system", "")

            # Conversation state, mirroring scripts/benchmark/reflect_eval_v2.py.
            conversation: Dict[str, Any] = {
                "messages": [],
                "cur_obs": pil_frames,
                "language_goal": goal_text,
                "vgoal_str": goal_text,
                "trajectory": [],
                "reflection": [],
                "thinking": [],
            }

            # Optional ECoT-style thinking pre-stage. When configured, the
            # thinking segment already injects `cur_obs`, so the subsequent
            # motion segment must not re-inject the observation window
            # (matches `add_image=(kind == "motion" and not is_thinking_in_plan)`
            # in scripts/benchmark/reflect_eval_v2.py).
            thinking_text = ""
            thinking_prompt = self.prompts.get("thinking")
            thinking_enabled = bool(thinking_prompt)
            if thinking_enabled:
                thinking_segment = build_thinking_segment(
                    conversation=conversation,
                    prompt_template=thinking_prompt,
                )
                messages = conversation["messages"] + thinking_segment
                raw_text = self._planner_model.generate(
                    messages,
                    model_role="motion",
                    output_format=OutputFormat.THINKING_V1.value,
                    instructions=system_prompt,
                )
                thinking_text = str(raw_text)
                conversation["messages"] = (
                    conversation["messages"]
                    + thinking_segment
                    + [ChatQuery("text", "assistant", raw_text)]
                )
                conversation["thinking"].append(
                    self._planner_model.parse_thinking_response(raw_text)
                )

            final_output: Optional[PlannerOutput] = None
            final_path_msg = Path()
            final_local_xyz = np.empty((0, 3), dtype=float)
            final_verdict = False
            final_reason = ""
            final_reflection_id = 0

            for refl in range(self._max_reflections + 1):
                # Default verdict True so that, when no critic is configured,
                # we publish the latest motion plan and break out after a
                # single iteration.
                iteration_verdict = True
                iteration_reason = ""
                iteration_critic_text = ""

                # Motion / motion_refine: always run.
                motion_kind = "motion" if refl == 0 else "motion_refine"
                motion_prompt = self.prompts.get(motion_kind)
                if not motion_prompt:
                    if motion_kind == "motion":
                        self.get_logger().warn(
                            "Missing 'motion' prompt; aborting plan."
                        )
                        break
                    self.get_logger().info(
                        f"No '{motion_kind}' prompt configured; "
                        "stopping refinement loop."
                    )
                    break
                motion_segment = build_motion_segment(
                    conversation=conversation,
                    prompt_template=motion_prompt,
                    add_image=(motion_kind == "motion" and not thinking_enabled),
                )
                messages = conversation["messages"] + motion_segment
                raw_text = self._planner_model.generate(
                    messages,
                    model_role="motion",
                    output_format=OutputFormat.TRAJECTORY_V1.value,
                    instructions=system_prompt,
                )
                output = self._planner_model.parse_motion_response(
                    raw_text,
                    image_np=base_image_np,
                )
                conversation["messages"] = (
                    conversation["messages"]
                    + motion_segment
                    + [ChatQuery("text", "assistant", output.raw_text)]
                )
                conversation["trajectory"].append(output.pixel_waypoints)

                path_msg, local_xyz = self._waypoints_to_path_msg(
                    output.xyz_waypoints, stamp=stamp, pose_msg=obs.pose_msg
                )
                final_output = output
                final_path_msg = path_msg
                final_local_xyz = local_xyz
                final_reflection_id = refl

                # Critic / critic_refine: optional; overrides verdict if run.
                critic_kind = "critic" if refl == 0 else "critic_refine"
                critic_prompt = self.prompts.get(critic_kind)
                if critic_prompt:
                    critic_segment = build_critic_segment(
                        conversation=conversation,
                        prompt_template=critic_prompt,
                        waypoints=output.pixel_waypoints,
                    )
                    messages = conversation["messages"] + critic_segment
                    raw_text = self._planner_model.generate(
                        messages,
                        model_role="critic",
                        output_format=OutputFormat.VERDICT_V1.value,
                        instructions=system_prompt,
                    )
                    critique = self._planner_model.parse_critic_response(raw_text)
                    conversation["messages"] = (
                        conversation["messages"]
                        + critic_segment
                        + [ChatQuery("text", "assistant", critique.raw_text)]
                    )
                    conversation["reflection"].append(
                        {"verdict": critique.verdict, "reason": critique.reason}
                    )
                    iteration_verdict = bool(critique.verdict)
                    iteration_reason = critique.reason
                    iteration_critic_text = str(critique.raw_text)
                    self.get_logger().info(
                        f"[refl={refl}] critic verdict={critique.verdict} "
                        f"reason={critique.reason}"
                    )

                final_verdict = iteration_verdict
                final_reason = iteration_reason

                # Publish artifacts after motion + (optional) critic, before
                # the conditional break. Attach the thinking trace only on
                # the first iteration so it renders once per goal.
                self._publish_iteration_artifacts(
                    base_image_np=base_image_np,
                    pixel_waypoints=output.pixel_waypoints,
                    local_xyz_waypoints=local_xyz,
                    header=header,
                    path_msg=path_msg,
                    reflection_id=refl,
                    verdict=iteration_verdict,
                    reason=iteration_reason,
                    thinking_text=thinking_text if refl == 0 else "",
                    motion_text=str(output.raw_text),
                    critic_text=iteration_critic_text,
                )

                if iteration_verdict:
                    break

            assert final_output is not None, "No assistant output from planner model."

            response.foresight_plan.header.stamp = stamp
            response.foresight_plan.header.frame_id = self._path_frame_id
            response.foresight_plan.path = final_path_msg
            response.foresight_plan.reflection_id = int(final_reflection_id)
            response.foresight_plan.verdict = Bool(data=bool(final_verdict))
            response.foresight_plan.reason = String(data=str(final_reason))

            if final_path_msg.poses:
                self.get_logger().info(
                    f"Final plan ({self._path_frame_id} frame, refl="
                    f"{final_reflection_id}, verdict={final_verdict}):"
                )
                for waypoint in final_path_msg.poses:
                    p = waypoint.pose.position
                    self.get_logger().info(f"  x={p.x:.3f} y={p.y:.3f} z={p.z:.3f}")
        except Exception as exc:
            import traceback
            self.get_logger().error(
                f"Planner request failed: {exc}\n{traceback.format_exc()}"
            )
            response.foresight_plan = ForesightPlannerMsg()
            response.foresight_plan.path = Path()
            response.foresight_plan.reflection_id = 0
            response.foresight_plan.verdict = Bool(data=False)
            response.foresight_plan.reason = String(data=f"planner exception: {exc}")
        return response


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WaypointPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
