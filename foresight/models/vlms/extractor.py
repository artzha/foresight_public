import re
import json
from typing import Optional

def extract_critic_trace(
    trace_str: str,
    *,
    default_verdict: int = 0,
    default_reason: str = "unparsed",
    max_reason_chars: int = 512,
):
    """
    Best-effort extraction of {"verdict": 0|1, "reason": "..."} from messy model text.

    Robustness features for common VLM formatting issues:
      - Strips ``` fences
      - Normalizes smart quotes (“ ” ‘ ’) to ASCII quotes
      - Tolerates outer wrapping quotes around the whole blob
      - Tolerates trailing junk after the JSON (e.g., …”}’)
      - Tolerates invalid escaping / truncated strings (falls back to raw substring)
      - Case-insensitive key matching for verdict/reason
      - Accepts verdict as 0/1, "0"/"1", true/false, yes/no, pass/fail
    """
    if trace_str is None:
        return None

    s = str(trace_str).strip()
    if not s:
        return json.dumps(
            {"verdict": int(default_verdict), "reason": default_reason[:max_reason_chars]},
            separators=(",", ":"),
            ensure_ascii=False,
        )

    # --- 1) Normalize / clean ---
    # Strip code fences
    s = re.sub(r"```(?:json|JSON)?", "", s)
    s = s.replace("```", "")

    # Normalize common unicode “smart quotes” and apostrophes
    s = s.translate(
        str.maketrans(
            {
                "\u201c": '"',  # “
                "\u201d": '"',  # ”
                "\u2018": "'",  # ‘
                "\u2019": "'",  # ’
                "\u2032": "'",  # ′
                "\u2033": '"',  # ″
            }
        )
    )

    # Escaped whitespace + real whitespace normalization
    s = s.replace("\\n", " ").replace("\\r", " ").replace("\\t", " ")
    s = re.sub(r"[\r\n\t]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # If the whole thing is wrapped in quotes, unwrap once (common in logs / reprs)
    if len(s) >= 2 and s[0] == s[-1] and s[0] in {"'", '"'}:
        inner = s[1:-1].strip()
        # Only unwrap if it looks like a wrapped JSON-ish blob
        if "verdict" in inner.lower() and ("{" in inner or ":" in inner):
            s = inner

    # If there's a JSON object embedded with trailing junk, prefer the widest {...} span.
    # (This handles …"}’ and …\\"}' cases.)
    if "{" in s and "}" in s:
        i = s.find("{")
        j = s.rfind("}")
        if i < j:
            s = s[i : j + 1]

    def _key_pat(word: str) -> str:
        # allow whitespace between letters (robust to odd wrapping)
        return r"\s*".join(map(re.escape, word))

    def _coerce_verdict(v: str) -> int:
        v2 = v.strip().strip('"').strip("'").strip().lower()

        if v2 in {"1", "true", "yes", "y", "pass", "passed", "ok"}:
            return 1
        if v2 in {"0", "false", "no", "n", "fail", "failed", "not_ok"}:
            return 0

        # Numeric-ish fallback
        m = re.search(r"[01]", v2)
        if m:
            return int(m.group(0))

        return int(default_verdict)

    def _clean_reason(r: str) -> str:
        r = str(r)
        r = r.translate(
            str.maketrans(
                {
                    "\u201c": '"',
                    "\u201d": '"',
                    "\u2018": "'",
                    "\u2019": "'",
                }
            )
        )
        r = r.replace("\\n", " ").replace("\\r", " ").replace("\\t", " ")
        r = re.sub(r"\s+", " ", r).strip()
        if not r:
            r = default_reason
        if len(r) > max_reason_chars:
            r = r[:max_reason_chars].rstrip()
        return r

    def _extract_value_after_key(text: str, key: str) -> Optional[str]:
        """
        Find <key> (case-insensitive) and extract the value after ':' or '='.
        If quoted, parse until an unescaped matching quote; if missing, read until end / close brace.
        If unquoted, read until comma / close brace / end.
        """
        key_re = re.compile(
            rf"""["']?\s*{_key_pat(key)}\s*["']?\s*(?:[:=])""",
            flags=re.IGNORECASE | re.VERBOSE,
        )
        m = key_re.search(text)
        if not m:
            return None

        k = m.end()
        # skip whitespace
        n = len(text)
        while k < n and text[k].isspace():
            k += 1
        if k >= n:
            return ""

        # If value starts with a quote, consume a quoted string (even if it's "broken")
        if text[k] in {'"', "'"}:
            quote = text[k]
            k += 1
            out = []
            escaped = False
            while k < n:
                ch = text[k]
                if escaped:
                    out.append(ch)
                    escaped = False
                else:
                    if ch == "\\":
                        escaped = True
                        # keep the backslash so the raw content remains meaningful
                        out.append(ch)
                    elif ch == quote:
                        # end of quoted string
                        return "".join(out)
                    else:
                        out.append(ch)
                k += 1

            # Unterminated quote: return what we have (best-effort)
            raw = "".join(out)
            # If it ends with a dangling backslash, drop it (helps cases like ...\\")
            if raw.endswith("\\"):
                raw = raw[:-1]
            return raw

        # Unquoted value: read until delimiter / end
        out = []
        while k < n:
            ch = text[k]
            if ch in {",", "}", "]"}:
                break
            out.append(ch)
            k += 1
        return "".join(out).strip()

    # --- 2) Extract verdict / reason (best-effort) ---
    v_raw = _extract_value_after_key(s, "verdict")
    verdict = _coerce_verdict(v_raw) if v_raw is not None else int(default_verdict)

    r_raw = _extract_value_after_key(s, "reason")
    reason = _clean_reason(r_raw if r_raw is not None else "")

    return json.dumps(
        {"verdict": int(verdict), "reason": reason},
        separators=(",", ":"),
        ensure_ascii=False,
    )

def extract_motion_trace(
    trace_str: str,
    *,
    min_points: int = 1,
    max_points: int = 50,
    clamp_01: bool = True,
    drop_out_of_bounds: bool = False,
):
    """
    Best-effort extraction of a motion trajectory from a messy string.

    Input notes:
      - The string may contain literal backslash-n sequences ("\\n") rather than real newlines.
      - The JSON may be truncated or otherwise invalid.

    Output:
      - JSON string: {"trajectory": [[x, y], ...]} (floats)
      - None if extraction fails (< min_points)

    Strategy:
      1) Normalize code-fences and convert literal "\\n"/"\\t"/"\\r" to whitespace.
      2) Locate "trajectory" key (robust to whitespace, but not required).
      3) Extract as many bracketed pairs [x, y] as possible from the tail.
         If none found, fallback to loose "x, y" pairs.
      4) Optional clamp/drop OOB and return canonical JSON.
    """
    if trace_str is None:
        return None
    s = str(trace_str)

    # 1) Normalize: remove ```json fences and make escaped newlines/tabs act like whitespace.
    #    Your snippets contain explicit "\\n" delimiters, so we turn them into spaces.
    s = re.sub(r"```(?:json|JSON)?", "", s)
    s = s.replace("```", "")
    s = s.replace("\\n", " ").replace("\\r", " ").replace("\\t", " ")

    # Floats like 0, 1, 0.5, .5, 1., -0.1, 1e-3, -2.3E+2
    FLOAT_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"

    # 2) Find "trajectory" key region (key letters may be separated by whitespace).
    #    This will still work even if the key got wrapped awkwardly, but in your case it won't.
    key_pat = r"t\s*r\s*a\s*j\s*e\s*c\s*t\s*o\s*r\s*y"
    m = re.search(rf'["\']?\s*{key_pat}\s*["\']?\s*:', s, flags=re.IGNORECASE)
    tail = s[m.end():] if m else s  # if key not found, scan whole string as last resort

    # 3) Extract bracketed pairs first: [x, y]
    pairs = []
    pair_pat = re.compile(rf"\[\s*({FLOAT_RE})\s*,\s*({FLOAT_RE})\s*\]", flags=re.IGNORECASE)

    for mm in pair_pat.finditer(tail):
        try:
            x = float(mm.group(1))
            y = float(mm.group(2))
        except ValueError:
            continue
        pairs.append((x, y))
        if len(pairs) >= max_points:
            break

    # 4) Fallback: loose x, y pairs if no bracketed pairs exist
    if not pairs:
        loose_pat = re.compile(rf"({FLOAT_RE})\s*,\s*({FLOAT_RE})", flags=re.IGNORECASE)
        for mm in loose_pat.finditer(tail):
            try:
                x = float(mm.group(1))
                y = float(mm.group(2))
            except ValueError:
                continue
            pairs.append((x, y))
            if len(pairs) >= max_points:
                break

    if len(pairs) < min_points:
        return None

    out = []
    for x, y in pairs:
        if clamp_01:
            x = 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
            y = 0.0 if y < 0.0 else (1.0 if y > 1.0 else y)

        if drop_out_of_bounds and not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            continue

        out.append([float(x), float(y)])

    if len(out) < min_points:
        return None

    # Canonical compact JSON
    return json.dumps({"trajectory": out}, separators=(",", ":"))