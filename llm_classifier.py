"""
llm_classifier.py
=================
Ollama client for exercise classification.

Key fixes:
- No num_predict cap: qwen3 internal thinking consumed all tokens, leaving
  none for JSON (done_reason=length). Removing the cap fixes empty responses.
- Checks done_reason=length and retries with a shorter summary if hit.
- think:false suppresses <think> tags for qwen3 (Ollama 0.7+).
- Auto-fallback from /api/chat to /api/generate for base models.
- Robust JSON extraction handles any extra prose around the JSON block.
"""

import json, os, re, threading, time
from dataclasses import dataclass, field
import requests

OLLAMA_URL   = os.environ.get("OLLAMA_URL",   "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")

SYSTEM_PROMPT = """\
You are a sports science AI. Classify exercises from joint-angle movement data.
You must respond with ONLY a JSON object — no explanation, no markdown, no thinking.

Given a movement summary, output exactly this JSON structure:
{
  "exercise":   "<one of: squats, lunges, push-ups, pull-ups, sit-ups, crunches, bicep curls, shoulder press, jumping jacks, burpees, mountain climbers, plank, unknown>",
  "rep_signal": "<one of: left_elbow, right_elbow, left_knee, right_knee, left_hip, right_hip, left_shoulder, right_shoulder, trunk, arm_spread, leg_spread>",
  "confidence": "<high, medium, or low>",
  "reasoning":  "<one short sentence>",
  "form_tips":  ["<tip1>", "<tip2>"]
}

Output ONLY the JSON object, nothing else.
"""

USER_TEMPLATE = "Movement data:\n{summary}\n\nOutput the JSON:"


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ClassificationResult:
    exercise:   str   = "waiting..."
    rep_signal: str   = ""
    confidence: str   = ""
    reasoning:  str   = ""
    form_tips:  list  = field(default_factory=list)
    error:      str   = ""
    timestamp:  float = field(default_factory=time.time)

    @property
    def is_valid(self):
        return not self.error and self.exercise not in ("waiting...", "")

    def display_lines(self):
        CYAN=(230,210,10); GREEN=(60,220,60); YELLOW=(0,200,230); GREY=(80,90,120)
        cc = {"high": GREEN, "medium": YELLOW, "low": GREY}.get(self.confidence, GREY)
        lines = [
            (self.exercise.upper().replace("-"," ").replace("_"," "), CYAN),
            ("Confidence: {}".format(self.confidence), cc),
        ]
        for tip in self.form_tips[:2]:
            lines.append(("> {}".format(tip), YELLOW))
        return lines


# ─────────────────────────────────────────────────────────────────────────────
# Text helpers
# ─────────────────────────────────────────────────────────────────────────────
def _strip_think(text):
    """Remove <think>...</think> blocks."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_json(text):
    """Extract first complete {...} JSON object from text, ignoring surrounding prose."""
    text = _strip_think(text)
    text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON found in:\n{}".format(text[:400]))
    depth, end = 0, -1
    for i, ch in enumerate(text[start:], start):
        if ch == "{":   depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == -1:
        raise ValueError("Unclosed JSON in:\n{}".format(text[start:start+400]))
    return json.loads(text[start:end])


def _is_qwen(model_name):
    return "qwen" in model_name.lower()


def _shorten_summary(summary):
    """
    Trim the summary to only the most informative lines.
    Keeps: header, top 5 joints by range, key observations, rep counts.
    Reduces input tokens so the model has more budget for JSON output.
    """
    lines        = summary.splitlines()
    keep         = []
    in_joints    = False
    joint_count  = 0
    in_repcounts = False
    rep_count    = 0

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Observation") or stripped.startswith("Most active"):
            keep.append(line)
        elif "Joint movements" in line:
            keep.append(line)
            in_joints = True
            joint_count = 0
        elif in_joints and line.startswith("  ") and joint_count < 5:
            keep.append(line)
            joint_count += 1
        elif in_joints and (not line.startswith("  ") or joint_count >= 5):
            in_joints = False
        if "Key observations" in line:
            keep.append(line)
        elif stripped.startswith("- ") or stripped.startswith("-"):
            keep.append(line)
        elif "Rep counts" in line:
            keep.append(line)
            in_repcounts = True
            rep_count = 0
        elif in_repcounts and line.startswith("  ") and rep_count < 5:
            keep.append(line)
            rep_count += 1

    # Deduplicate while preserving order
    seen, result = set(), []
    for line in keep:
        if line not in seen:
            seen.add(line)
            result.append(line)
    return "\n".join(result)


# ─────────────────────────────────────────────────────────────────────────────
# Classifier
# ─────────────────────────────────────────────────────────────────────────────
class LLMClassifier:
    def __init__(self, model=OLLAMA_MODEL, url=OLLAMA_URL):
        self.model     = model
        self.url       = url
        self.result    = ClassificationResult()
        self._lock     = threading.Lock()
        self._busy     = False
        self._use_chat = True

    @property
    def busy(self):
        return self._busy

    def request(self, summary):
        with self._lock:
            if self._busy:
                return
            self._busy = True
        threading.Thread(target=self._call, args=(summary,), daemon=True).start()

    def _call(self, summary):
        try:
            result = self._query(summary)
        except Exception as e:
            result = ClassificationResult(error=str(e))
            print("  [LLM error]", e)
        with self._lock:
            self.result = result
            self._busy  = False

    # ── routing ───────────────────────────────────────────────────────────────
    def _query(self, summary):
        if self._use_chat:
            try:
                return self._chat(summary)
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 404:
                    print("  [LLM] /api/chat not available, switching to /api/generate")
                    self._use_chat = False
                else:
                    raise
        return self._generate(summary)

    # ── /api/chat ─────────────────────────────────────────────────────────────
    def _chat(self, summary):
        payload = {
            "model":  self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": USER_TEMPLATE.format(summary=summary)},
            ],
            "options": self._options(),
        }
        resp = requests.post("{}/api/chat".format(self.url), json=payload, timeout=120)
        resp.raise_for_status()
        data        = resp.json()
        content     = data.get("message", {}).get("content", "")
        done_reason = data.get("done_reason", "")
        return self._parse(content, done_reason, summary, via="chat")

    # ── /api/generate ─────────────────────────────────────────────────────────
    def _generate(self, summary):
        prompt = "{}\n\n{}".format(SYSTEM_PROMPT, USER_TEMPLATE.format(summary=summary))
        payload = {
            "model":  self.model,
            "prompt": prompt,
            "stream": False,
            "options": self._options(),
        }
        resp = requests.post("{}/api/generate".format(self.url), json=payload, timeout=120)
        resp.raise_for_status()
        data        = resp.json()
        content     = data.get("response", "")
        done_reason = data.get("done_reason", "")
        return self._parse(content, done_reason, summary, via="generate")

    # ── parse ─────────────────────────────────────────────────────────────────
    def _parse(self, content, done_reason, summary, via):
        # done_reason='length' = hit token limit, output was cut — retry shorter
        if done_reason == "length":
            print("  [LLM] Token limit hit (done_reason=length). "
                  "Retrying with condensed summary...")
            short = _shorten_summary(summary)
            # Avoid infinite retry: if already short, just fail gracefully
            if len(short) >= len(summary) * 0.9:
                return ClassificationResult(
                    error="Token limit hit even on short summary. "
                          "Try a smaller model or increase context.")
            return self._chat(short) if via == "chat" else self._generate(short)

        stripped = _strip_think(content)
        if not stripped or len(stripped) < 5:
            print("  [LLM] Empty response (think suppressed all output). "
                  "Retrying with condensed summary...")
            short = _shorten_summary(summary)
            if len(short) >= len(summary) * 0.9:
                return ClassificationResult(error="Empty response even after shortening.")
            return self._chat(short) if via == "chat" else self._generate(short)

        parsed = _extract_json(stripped)
        result = ClassificationResult(
            exercise   = str(parsed.get("exercise",   "unknown")),
            rep_signal = str(parsed.get("rep_signal", "")),
            confidence = str(parsed.get("confidence", "low")),
            reasoning  = str(parsed.get("reasoning",  "")),
            form_tips  = list(parsed.get("form_tips",  [])),
        )
        print("  [LLM] -> {} ({}) via {}  rep_signal={}".format(
            result.exercise, result.confidence, via, result.rep_signal))
        return result

    # ── options ───────────────────────────────────────────────────────────────
    def _options(self):
        # IMPORTANT: Do NOT set num_predict for qwen3.
        # qwen3 uses tokens for internal reasoning even when think:false.
        # Capping at 400 left zero tokens for JSON output (done_reason=length).
        # Without a cap, the model finishes naturally (~150 tokens for JSON).
        opts = {"temperature": 0.1}
        if _is_qwen(self.model):
            opts["think"] = False   # suppress <think> output (Ollama 0.7+)
        return opts

    # ── health check ──────────────────────────────────────────────────────────
    def check_connection(self):
        try:
            r = requests.get("{}/api/tags".format(self.url), timeout=3)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            if not models:
                return False, "Ollama running but no models. Run: ollama pull llama3"
            base = self.model.split(":")[0]
            if not any(base in m for m in models):
                return True, "WARNING: '{}' not found. Available: {}".format(
                    self.model, ", ".join(models))
            return True, "Ollama OK  ({})".format(self.model)
        except requests.ConnectionError:
            return False, "Cannot reach Ollama at {}. Run: ollama serve".format(self.url)
        except Exception as e:
            return False, "Ollama error: {}".format(e)