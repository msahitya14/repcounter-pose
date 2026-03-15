"""
llm_classifier.py
=================
Sends movement summaries to Ollama and parses exercise classification +
form feedback. Triggered by joint-activity fingerprint changes.
"""

import json, os, re, threading, time
from dataclasses import dataclass, field
import requests

OLLAMA_URL   = os.environ.get("OLLAMA_URL",   "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")

SYSTEM_PROMPT = """\
You are a sports science AI that classifies exercises from joint-angle \
movement data captured by a pose-estimation camera.

You will receive a movement summary describing which joints moved, by how \
much, and how many rep cycles were detected.

Your job:
1. Identify the most likely exercise from this list:
   squats, lunges, push-ups, pull-ups, sit-ups, crunches,
   bicep curls, shoulder press, jumping jacks, burpees,
   mountain climbers, plank, unknown
2. Name the single best joint angle signal to use as the rep counter for \
   that exercise (must be one of: left_elbow, right_elbow, left_knee, \
   right_knee, left_hip, right_hip, left_shoulder, right_shoulder, \
   trunk, arm_spread, leg_spread).
3. Give a confidence: high / medium / low.
4. Give 1-3 short, specific form-feedback tips based on the angle data.

Respond ONLY with valid JSON, no markdown, no extra text:
{
  "exercise":   "<name>",
  "rep_signal": "<joint_angle_name>",
  "confidence": "<high|medium|low>",
  "reasoning":  "<one sentence>",
  "form_tips":  ["<tip1>", "<tip2>"]
}
"""

USER_TEMPLATE = """\
Movement summary:

{summary}

Classify the exercise and identify the best rep-counting signal.
"""


@dataclass
class ClassificationResult:
    exercise:   str  = "waiting..."
    rep_signal: str  = ""
    confidence: str  = ""
    reasoning:  str  = ""
    form_tips:  list = field(default_factory=list)
    error:      str  = ""
    timestamp:  float = field(default_factory=time.time)

    @property
    def is_valid(self):
        return not self.error and self.exercise not in ("waiting...", "")

    def display_lines(self):
        CYAN   = (230, 210,  10)
        GREEN  = ( 60, 220,  60)
        YELLOW = (  0, 200, 230)
        GREY   = ( 80,  90, 120)
        conf_color = {"high": GREEN, "medium": YELLOW, "low": GREY}.get(
            self.confidence, GREY)
        lines = [
            (self.exercise.upper().replace("-", " ").replace("_", " "), CYAN),
            ("Confidence: {}".format(self.confidence), conf_color),
        ]
        for tip in self.form_tips[:2]:
            lines.append(("> {}".format(tip), YELLOW))
        return lines


class LLMClassifier:
    def __init__(self, model=OLLAMA_MODEL, url=OLLAMA_URL):
        self.model  = model
        self.url    = url
        self.result = ClassificationResult()
        self._lock  = threading.Lock()
        self._busy  = False

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

    def _query(self, summary):
        payload = {
            "model":  self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": USER_TEMPLATE.format(summary=summary)},
            ],
            "options": {"temperature": 0.1, "num_predict": 300},
        }
        resp = requests.post("{}/api/chat".format(self.url),
                             json=payload, timeout=30)
        resp.raise_for_status()
        content = resp.json()["message"]["content"].strip()
        clean   = re.sub(r"```[a-z]*", "", content).replace("```", "").strip()
        # Strip <think>...</think> tags some models (qwen3) emit
        clean   = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL).strip()
        parsed  = json.loads(clean)
        return ClassificationResult(
            exercise   = parsed.get("exercise",   "unknown"),
            rep_signal = parsed.get("rep_signal", ""),
            confidence = parsed.get("confidence", "low"),
            reasoning  = parsed.get("reasoning",  ""),
            form_tips  = parsed.get("form_tips",  []),
        )

    def check_connection(self):
        try:
            r = requests.get("{}/api/tags".format(self.url), timeout=3)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            if not any(self.model in m for m in models):
                available = ", ".join(models)
                return True, "Model '{}' not found. Available: {}".format(
                    self.model, available)
            return True, "Ollama OK  ({})".format(self.model)
        except requests.ConnectionError:
            return False, "Cannot reach Ollama at {}. Run: ollama serve".format(self.url)
        except Exception as e:
            return False, "Ollama error: {}".format(e)
