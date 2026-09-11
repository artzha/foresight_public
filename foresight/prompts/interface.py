# interface.py
from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Type, Protocol, runtime_checkable, Optional
from pydantic import BaseModel, Field, conint, confloat, conlist, constr, field_validator
import json
import re

from foresight.models.vlms.extractor import extract_motion_trace
from foresight.core.constants import VALID_ENVIRONMENTS

# ---------- Types ----------
Point = conlist(confloat(ge=0.0, le=1.0), min_length=2, max_length=2)

class ContentType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"

class Role(str, Enum):
    USER = "user"
    DEVELOPER = "developer"
    SYSTEM = "system"
    ASSISTANT = "assistant"

@dataclass
class ChatQuery:
    type: str = ContentType.TEXT.value
    role: str = Role.USER.value
    # text  -> str
    # image -> PIL.Image.Image | path/URI str
    # video -> list[PIL.Image.Image | path/URI str]
    content: Any = field(default_factory=str)

    def __post_init__(self) -> None:
        # leave content as-is; caller guarantees correct type for each role/type
        pass

    def as_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "role": self.role, "content": self.content}

# ----- Legacy (kept; now behind DecisionsPayload) -----
class ChoiceReason(BaseModel):
    choice: conint(ge=0, le=9) = Field(description="Index of the path choice.")
    reason: constr(min_length=3, max_length=512) = Field(description="Brief rationale for this choice.")

class ReasoningTrace(BaseModel):
    decisions: List[ChoiceReason] = Field(
        description="A list of decisions, where each entry contains the chosen path index and its rationale."
    )

# ----- Task formats -----
class OutputFormat(str, Enum):
    DECISIONS_V1 = "decisions:v1"     # {"decisions":[{choice,reason},...]}
    TRAJECTORY_V1 = "trajectory:v1"   # {"trajectory":[[x,y],...]}
    VERDICT_V1   = "verdict:v1"       # {"verdict":"0|1","reason":"..."}
    VERDICT_V2   = "verdict:v2"       # {"verdict":"0|1","reason":"...","correction":"..."}
    REWARD_V1 = "reward:v1"           # {"reward": float, "reason": "..."}
    GROUP_CRITIQUE_V1 = "group_critique:v1"  # {"critiques":[{"idx":1,"verdict":0|1,"reason":"..."}]}
    DATAGEN_GOAL_V1 = "datagen:goal:v1"  # custom format for datagen motion goal tasks
    THINKING_V1 = "thinking:v1"          # ECoT chain-of-causation reasoning trace

@dataclass
class TaskSpec:
    id: str
    system_prompt: str
    meta: Dict[str, Any] = field(default_factory=dict)

def _coerce_json(raw: Any) -> Any:
    if isinstance(raw, BaseModel):
        return raw.model_dump()
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return raw
    return raw

def extract_json_like(text: str) -> str:
    """
    Best-effort extractor for a JSON object/array from model text.
    - Strips ``` fences (with/without 'json')
    - If multiple braces exist, grabs the last valid JSON object/array
    """
    s = text.strip()
    # remove code fences if present
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", s, re.DOTALL | re.IGNORECASE)
    if fence:
        s = fence.group(1).strip()

    # if still not valid JSON, try to find last {...} or [...]
    def try_load(candidate: str) -> Optional[str]:
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            return None

    ok = try_load(s)
    if ok is not None:
        return s

    # Greedy scan from end for {...}
    last_obj = None
    for m in re.finditer(r"(\{.*\})", s, re.DOTALL):
        cand = m.group(1)
        if try_load(cand):
            last_obj = cand
    if last_obj:
        return last_obj

    # Try array
    last_arr = None
    for m in re.finditer(r"(\[.*\])", s, re.DOTALL):
        cand = m.group(1)
        if try_load(cand):
            last_arr = cand
    if last_arr:
        return last_arr

    # Give up; return as-is
    return s

# ---------- Payload protocol ----------
@runtime_checkable
class BasePayload(Protocol):
    @classmethod
    def parse_payload(cls, raw: Any) -> "BasePayload": ...
    def to_unified(self, meta: Dict[str, Any]) -> Dict[str, Any]: ...
    @classmethod
    def json_schema_str(cls) -> str: ...

# ---------- Payloads ----------
class DecisionsPayload(ReasoningTrace):
    @classmethod
    def parse_payload(cls, raw: Any) -> "DecisionsPayload":
        x = _coerce_json(raw)
        if isinstance(x, dict) and "decision" in x and "decisions" not in x:
            x = {"decisions": [x["decision"]]}
        if isinstance(x, dict) and "decisions" not in x:
            x = {"decisions": [x]}  # coerce single -> list
        return cls(**x)

    def to_unified(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        final = self.decisions[-1]
        inter = self.decisions[:-1]
        return {
            **meta,
            "intermediate_responses": [
                {"stage": i, "choice": d.choice, "reason": d.reason} for i, d in enumerate(inter)
            ],
            "final_response": {"stage": len(inter), "choice": final.choice, "reason": final.reason},
        }

    @classmethod
    def json_schema_str(cls) -> str:
        return json.dumps(cls.model_json_schema(), ensure_ascii=False)

class TrajectoryPayload(BaseModel):
    trajectory: conlist(Point, min_length=1) = Field(
        description="Normalized pixel coordinates as [[x,y], ...], each in [0,1]."
    )

    @field_validator("trajectory")
    @classmethod
    def _unit_square(cls, v):
        for x, y in v:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError("coords must be normalized to [0,1]")
        return v

    @classmethod
    def parse_payload(cls, raw: Any) -> "TrajectoryPayload":
        """
        Accepts:
          - dict with 'trajectory': {...}
          - dict with 'points': {...} (legacy)
          - bare 2D list/tuple: [[x, y], ...]
        """
        x = _coerce_json(raw)

        # Case 1: dict input
        if isinstance(x, dict):
            if "trajectory" in x:
                # already in the correct shape
                return cls(**x)
            if "points" in x:
                # legacy format: {"points": [[x, y], ...]}
                return cls(trajectory=x["points"])
            # fall through: unrecognized dict
            raise ValueError(
                "Expected dict with key 'trajectory' or 'points' for TrajectoryPayload"
            )

        # Case 2: bare list/tuple → treat as trajectory directly
        if isinstance(x, (list, tuple)):
            return cls(trajectory=x)
        
        if isinstance(x, str):
            x = json.loads(extract_motion_trace(x))
            return cls(trajectory=x.get('trajectory', []))

        # Anything else is unsupported
        raise TypeError(
            f"Cannot parse TrajectoryPayload from type {type(x)!r}; "
            "expected dict or 2D list/tuple"
        )

    def to_unified(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        return {**meta, "trajectory": self.trajectory}

    @classmethod
    def json_schema_str(cls) -> str:
        return json.dumps(cls.model_json_schema(), ensure_ascii=False)

class VerdictPayload(BaseModel):
    verdict: conint(ge=0, le=1) = Field(description="Binary verdict: 0 or 1")
    reason: constr(min_length=1, max_length=1024)

    @classmethod
    def parse_payload(cls, raw: Any) -> "VerdictPayload":
        x = _coerce_json(raw)

        if isinstance(x, dict):
            if "verdict" in x and "reason" in x:
                return cls(**x)
            raise ValueError(
                "Expected dict with keys 'verdict' and 'reason' for VerdictPayload" 
            )
        
        if isinstance(x, str):
            x = json.loads(extract_motion_trace(x))
            return cls(**x)
        
        raise TypeError(
            f"Cannot parse VerdictPayload from type {type(x)!r}; expected dict or str"
        )

    def to_unified(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        return {**meta, "verdict": int(self.verdict), "reason": self.reason}

    @classmethod
    def json_schema_str(cls) -> str:
        return json.dumps(cls.model_json_schema(), ensure_ascii=False)

class VerdictPayloadv2(BaseModel):
    verdict: conint(ge=0, le=1) = Field(description="Binary verdict: 0 or 1")
    reason: constr(min_length=1, max_length=256) = Field(
        description="Short image-grounded reason for the verdict."
    )
    correction: constr(min_length=1, max_length=256) = Field(
        description="Exact correction to the path."
    )

    @classmethod
    def parse_payload(cls, raw: Any) -> "VerdictPayloadv2":
        x = _coerce_json(raw)
        if not isinstance(x, dict):
            raise TypeError(f"Cannot parse VerdictPayloadv2 from type {type(x)!r}; expected dict")

        # verdict is often returned as "0"/"1"
        v = x.get("verdict", 0)
        if isinstance(v, str):
            v2 = v.strip()
            if v2 in {"0", "1"}:
                x["verdict"] = int(v2)

        return cls(**x)

    def to_unified(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        return {
            **meta,
            "verdict": int(self.verdict),
            "reason": self.reason,
            "correction": self.correction,
        }

    @classmethod
    def json_schema_str(cls) -> str:
        return json.dumps(cls.model_json_schema(), ensure_ascii=False)


class RewardPayload(BaseModel):
    reward: confloat(ge=-1000.0, le=1000.0) = Field(
        description="Scalar reward score in [-1000, 1000]."
    )

    @classmethod
    def parse_payload(cls, raw: Any) -> "RewardPayload":
        x = _coerce_json(raw)

        if isinstance(x, dict):
            if "reward" in x:
                try:
                    x["reward"] = float(x["reward"])
                except (TypeError, ValueError):
                    pass
                return cls(reward=x["reward"])
            raise ValueError("Expected dict with key 'reward' for RewardPayload")

        if isinstance(x, str):
            # Allow simple scalar strings as a fallback (e.g., "0.37").
            try:
                return cls(reward=float(x.strip()))
            except ValueError as exc:
                raise TypeError(
                    f"Cannot parse RewardPayload from string: {x!r}; expected JSON with 'reward' or scalar float string"
                ) from exc

        raise TypeError(
            f"Cannot parse RewardPayload from type {type(x)!r}; expected dict or str"
        )

    def to_unified(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        return {"reward": float(self.reward)}

    @classmethod
    def json_schema_str(cls) -> str:
        return json.dumps(cls.model_json_schema(), ensure_ascii=False)


class GroupCritiqueItem(BaseModel):
    idx: conint(ge=1, le=32) = Field(description="1-based rollout index in the overlay.")
    verdict: conint(ge=0, le=1) = Field(description="Binary verdict for this indexed rollout.")
    reason: constr(min_length=0, max_length=512) = Field(description="Short image-grounded reason.")

    @field_validator("reason", mode="before")
    @classmethod
    def _normalize_reason(cls, v):
        if v is None:
            return ""
        return str(v)


class GroupCritiquePayload(BaseModel):
    critiques: List[GroupCritiqueItem] = Field(
        description="Per-index critiques for grouped rollout overlay."
    )

    @classmethod
    def parse_payload(cls, raw: Any) -> "GroupCritiquePayload":
        x = _coerce_json(raw)
        if isinstance(x, str):
            try:
                x = json.loads(extract_json_like(x))
            except Exception:
                # Be fault-tolerant to truncated/invalid model text in long runs.
                print("Warning: GroupCritiquePayload received non-JSON string; defaulting to empty critiques.")
                return cls(critiques=[])
        if isinstance(x, dict):
            if "critiques" in x:
                return cls(**x)
            if "decisions" in x:
                mapped = []
                for d in x.get("decisions", []):
                    if not isinstance(d, dict):
                        continue
                    mapped.append(
                        {
                            "idx": d.get("idx", d.get("choice", 1)),
                            "verdict": d.get("verdict", 1),
                            "reason": d.get("reason", ""),
                        }
                    )
                return cls(critiques=mapped)
        if isinstance(x, list):
            return cls(critiques=x)
        raise TypeError(
            f"Cannot parse GroupCritiquePayload from type {type(x)!r}; expected dict or list"
        )

    def to_unified(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        return {
            **meta,
            "critiques": [
                {"idx": int(item.idx), "verdict": int(item.verdict), "reason": str(item.reason)}
                for item in self.critiques
            ],
        }

    @classmethod
    def json_schema_str(cls) -> str:
        return json.dumps(cls.model_json_schema(), ensure_ascii=False)


class DatagenGoalPayload(BaseModel):
    language_goal: List[constr(min_length=3, max_length=128)] = Field(
        description=(
            "Short phrase describing the final goal of the robot."
        )
    )
    environment_name: Optional[str] = Field(
        default=None,
        description="Environment class for the observation."
    )

    @field_validator("environment_name")
    @classmethod
    def _valid_environment_name(cls, v):
        if v is None:
            return None
        env = str(v).strip().lower()
        if env not in VALID_ENVIRONMENTS:
            raise ValueError(
                f"environment_name must be one of {sorted(VALID_ENVIRONMENTS)}, got: {v}"
            )
        return env

    @classmethod
    def parse_payload(cls, raw: Any) -> "DatagenGoalPayload":
        """
        Normalize common variants from the model output and construct a DatagenGoalPayload.

        - Accepts `language_goal` as either a string or a list of strings.
        - Keeps backward compatibility with legacy outputs.
        """
        x = _coerce_json(raw)

        if not isinstance(x, dict):
            return cls(language_goal=[str(x)])

        if "environment_name" not in x:
            if "env_class" in x:
                x["environment_name"] = x["env_class"]
            elif "environment" in x:
                x["environment_name"] = x["environment"]

        if "language_goal" in x:
            goals = x["language_goal"]
            if isinstance(goals, str):
                x["language_goal"] = [goals]
            elif isinstance(goals, list):
                x["language_goal"] = [str(goal) for goal in goals if str(goal).strip()]
            else:
                x["language_goal"] = [str(goals)]
            return cls(**x)

        if "subgoal" in x:
            subgoal = x["subgoal"]
            if isinstance(subgoal, str):
                x["language_goal"] = [subgoal]
            elif isinstance(subgoal, list):
                x["language_goal"] = [str(goal) for goal in subgoal if str(goal).strip()]
            else:
                x["language_goal"] = [str(subgoal)]
            return cls(**x)

        # Fallback for other dictionary structures by serializing them.
        return cls(language_goal=[json.dumps(x, ensure_ascii=False)])

    def to_unified(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        return {
            **meta,
            "language_goal": self.language_goal,
            "environment_name": self.environment_name,
        }

    @classmethod
    def json_schema_str(cls) -> str:
        return json.dumps(cls.model_json_schema(), ensure_ascii=False)

class ThinkingEntity(BaseModel):
    name: constr(min_length=1, max_length=64) = Field(
        description="snake_case noun naming a visible entity."
    )
    location: conlist(confloat(ge=0.0, le=1.0), min_length=2, max_length=2) = Field(
        description="Normalized [x, y] pixel centroid of the entity, each in [0, 1]."
    )


class ThinkingMetaAction(BaseModel):
    longitudinal: constr(min_length=1, max_length=32) = Field(
        description="One of: stop, slow, maintain, accelerate."
    )
    lateral: constr(min_length=1, max_length=32) = Field(
        description="One of: keep, turn_left, turn_right, sidestep_left, sidestep_right."
    )


# Per-field truncation limits for ThinkingPayload.parse_payload (must match the
# constr(max_length=...) declarations on ThinkingPayload fields).
_THINKING_TRUNCATE: Dict[str, int] = {
    "scene_caption": 128,
    "spatial_reasoning": 128,
    "rationale": 128,
}
_THINKING_ENTITY_NAME_MAX: int = 64
_THINKING_CLUE_VALUE_MAX: int = 128


class ThinkingPayload(BaseModel):
    """ECoT chain-of-causation reasoning trace produced from a current observation.

    Mirrors the six-key JSON schema described in foresight/prompts/ecot/thinking_prompt.txt.
    """
    scene_caption: constr(min_length=1, max_length=128)
    entities: List[ThinkingEntity] = Field(min_length=1, max_length=3)
    clues: Dict[str, constr(min_length=1, max_length=128)] = Field(
        description="Causal-role map: entity name -> short role description (1-3 entries)."
    )
    spatial_reasoning: constr(min_length=1, max_length=128)
    meta_action: ThinkingMetaAction
    rationale: constr(min_length=1, max_length=128)

    # Soft-buffer limits: strings exceeding these are silently truncated in
    # parse_payload rather than raising a validation error.  Values match the
    # constr(max_length=...) declarations on the fields above.

    @classmethod
    def parse_payload(cls, raw: Any) -> "ThinkingPayload":
        x = _coerce_json(raw)
        if isinstance(x, str):
            try:
                x = json.loads(extract_json_like(x))
            except Exception as exc:
                raise TypeError(
                    f"Cannot parse ThinkingPayload from string: {raw!r}"
                ) from exc
        if not isinstance(x, dict):
            raise TypeError(
                f"Cannot parse ThinkingPayload from type {type(x)!r}; expected dict"
            )

        # Silently truncate top-level string fields that exceed the Pydantic limit.
        for field, limit in _THINKING_TRUNCATE.items():
            if isinstance(x.get(field), str) and len(x[field]) > limit:
                x[field] = x[field][:limit]

        # Truncate entity names.
        if isinstance(x.get("entities"), list):
            for ent in x["entities"]:
                if isinstance(ent, dict) and isinstance(ent.get("name"), str):
                    if len(ent["name"]) > _THINKING_ENTITY_NAME_MAX:
                        ent["name"] = ent["name"][:_THINKING_ENTITY_NAME_MAX]

        # Tolerate clues returned as a list of {name, role} pairs; also truncate values.
        clues = x.get("clues")
        if isinstance(clues, list):
            mapped: Dict[str, str] = {}
            for item in clues:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "")).strip()
                role = str(item.get("role", item.get("description", ""))).strip()
                if name and role:
                    mapped[name] = role
            x["clues"] = mapped
        if isinstance(x.get("clues"), dict):
            x["clues"] = {
                k: v[:_THINKING_CLUE_VALUE_MAX] if isinstance(v, str) and len(v) > _THINKING_CLUE_VALUE_MAX else v
                for k, v in x["clues"].items()
            }

        return cls(**x)

    def to_unified(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        return {
            **meta,
            "scene_caption": self.scene_caption,
            "entities": [
                {"name": e.name, "location": [float(e.location[0]), float(e.location[1])]}
                for e in self.entities
            ],
            "clues": dict(self.clues),
            "spatial_reasoning": self.spatial_reasoning,
            "meta_action": {
                "longitudinal": self.meta_action.longitudinal,
                "lateral": self.meta_action.lateral,
            },
            "rationale": self.rationale,
        }

    @classmethod
    def model_json_schema(cls, **_kwargs) -> Dict[str, Any]:
        """Return a flat, Gemini-compatible JSON schema with no $defs or $ref.

        Pydantic's default schema for this model uses $defs/$ref for the nested
        ThinkingEntity and ThinkingMetaAction models, and additionalProperties
        for the clues dict — both rejected by the Gemini structured-output API.
        This hand-written override is equivalent in meaning but fully inlined.

        maxLength values here are the *target* the model is asked to respect.
        parse_payload enforces a soft buffer on top: strings up to the Pydantic
        constr limit are accepted and silently truncated rather than rejected.
        """
        return {
            "type": "object",
            "properties": {
                "scene_caption": {
                    "type": "string",
                    "maxLength": 128,
                    "description": (
                        "Short caption (≤128 chars) describing traversable surfaces, doorways, "
                        "dynamic obstacles, and lighting in the current observation."
                    ),
                },
                "entities": {
                    "type": "array",
                    "description": "Grounded list of up to three distinct visual entities visible in the current image.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "maxLength": 64,
                                "description": "snake_case noun naming a visible entity (≤64 chars).",
                            },
                            "location": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "Normalized [x, y] pixel centroid of the entity, each in [0, 1].",
                            },
                        },
                        "required": ["name", "location"],
                    },
                },
                "clues": {
                    "type": "object",
                    "description": (
                        "Chain-of-causation map: entity name -> short causal role description (≤128 chars each). "
                        "Keys MUST match names from entities. Up to three entries."
                    ),
                },
                "spatial_reasoning": {
                    "type": "string",
                    "maxLength": 128,
                    "description": "One short sentence (≤128 chars) explaining the spatial relationship the agent must respect.",
                },
                "meta_action": {
                    "type": "object",
                    "properties": {
                        "longitudinal": {
                            "type": "string",
                            "maxLength": 32,
                            "description": "One of: stop, slow, maintain, accelerate.",
                        },
                        "lateral": {
                            "type": "string",
                            "maxLength": 32,
                            "description": "One of: keep, turn_left, turn_right, sidestep_left, sidestep_right.",
                        },
                    },
                    "required": ["longitudinal", "lateral"],
                },
                "rationale": {
                    "type": "string",
                    "maxLength": 128,
                    "description": "One sentence (≤128 chars) linking the selected clues to the chosen meta_action and language instruction.",
                },
            },
            "required": ["scene_caption", "entities", "clues", "spatial_reasoning", "meta_action", "rationale"],
        }

    @classmethod
    def json_schema_str(cls) -> str:
        return json.dumps(cls.model_json_schema(), ensure_ascii=False)

FORMAT_REGISTRY: dict[OutputFormat, Type[BaseModel]] = {
    OutputFormat.DECISIONS_V1: DecisionsPayload,
    OutputFormat.TRAJECTORY_V1: TrajectoryPayload,
    OutputFormat.VERDICT_V1:   VerdictPayload,
    OutputFormat.VERDICT_V2:   VerdictPayloadv2,
    OutputFormat.REWARD_V1: RewardPayload,
    OutputFormat.GROUP_CRITIQUE_V1: GroupCritiquePayload,
    OutputFormat.DATAGEN_GOAL_V1: DatagenGoalPayload,
    OutputFormat.THINKING_V1: ThinkingPayload,
}

def schema_for(fmt: OutputFormat, return_cls=False) -> str:
    cls = FORMAT_REGISTRY[fmt]
    if return_cls:
        return cls
    # type: ignore[attr-defined]
    return cls.json_schema_str()  # every payload implements json_schema_str

# ---------- Unified envelope returned by *every* model ----------
@dataclass
class UnifiedEnvelope:
    model_name: str
    output_format: OutputFormat
    unified: Dict[str, Any]
    parsed: BaseModel | None
    raw_text: str
    usage: Dict[str, Any]   # keep it a plain dict for portability
    meta: Dict[str, Any] = field(default_factory=dict)

def parse_and_unify(
    raw_text_or_json: Any,
    fmt: OutputFormat,
    *,
    meta: Dict[str, Any] | None = None,
    model_name: str = "unknown",
    usage: Dict[str, Any] | None = None,
    verbosity: int = 0,
) -> UnifiedEnvelope:
    cls = FORMAT_REGISTRY[fmt]
    raw_json = raw_text_or_json
    raw_text = ""

    if isinstance(raw_text_or_json, str):
        raw_text = raw_text_or_json
        candidate = extract_json_like(raw_text_or_json)
        try:
            raw_json = json.loads(candidate)
        except Exception:
            # as a last resort, pass the string straight to payload (some payloads accept strings)
            raw_json = raw_text_or_json
            if verbosity >= 1:
                print(f"Warning: failed to parse JSON from model output; passing raw text to payload: {raw_text_or_json!r}")

    # type: ignore[attr-defined]
    parsed = cls.parse_payload(raw_json)  # every payload implements parse_payload
    # type: ignore[attr-defined]
    unified = parsed.to_unified(meta or {})

    return UnifiedEnvelope(
        model_name=model_name,
        output_format=fmt,
        unified=unified,
        parsed=parsed,      # keep parsed Pydantic for downstream programmatic access
        raw_text=raw_text if isinstance(raw_text_or_json, str) else json.dumps(raw_json),
        usage=usage or {},
        meta=meta or {},
    )
