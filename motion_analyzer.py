"""
motion_analyzer.py
==================
Tracks joint angles, counts reps via zero-crossing detection on the
velocity signal (more robust than peak detection for all exercise types),
detects exercise changes via joint-activity fingerprinting, and produces
natural-language summaries for the LLM.
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List

# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe BlazePose 33-point landmark indices
# ─────────────────────────────────────────────────────────────────────────────
LM = dict(
    nose=0,
    left_shoulder=11,  right_shoulder=12,
    left_elbow=13,     right_elbow=14,
    left_wrist=15,     right_wrist=16,
    left_hip=23,       right_hip=24,
    left_knee=25,      right_knee=26,
    left_ankle=27,     right_ankle=28,
    left_heel=29,      right_heel=30,
    left_foot=31,      right_foot=32,
)

# ─────────────────────────────────────────────────────────────────────────────
# Joint angle definitions  (name, idx_a, idx_vertex, idx_c)
# ─────────────────────────────────────────────────────────────────────────────
ANGLE_DEFS = [
    ("left_elbow",      11, 13, 15),
    ("right_elbow",     12, 14, 16),
    ("left_knee",       23, 25, 27),
    ("right_knee",      24, 26, 28),
    ("left_hip",        11, 23, 25),
    ("right_hip",       12, 24, 26),
    ("left_shoulder",   13, 11, 23),
    ("right_shoulder",  14, 12, 24),
    ("trunk",           25, 23, 11),
    ("arm_spread",      15, 11, 16),
    ("leg_spread",      27, 23, 28),
    ("neck",             0, 11, 12),
]

# ─────────────────────────────────────────────────────────────────────────────
# Exercise-to-primary-signal mapping
# Which angle is the BEST rep signal for each exercise (after LLM classifies).
# ─────────────────────────────────────────────────────────────────────────────
EXERCISE_SIGNAL = {
    "squats":          "left_knee",
    "lunges":          "left_knee",
    "push-ups":        "left_elbow",
    "pull-ups":        "left_elbow",
    "bicep curls":     "left_elbow",
    "shoulder press":  "left_shoulder",
    "sit-ups":         "trunk",
    "crunches":        "trunk",
    "jumping jacks":   "arm_spread",
    "burpees":         "left_knee",
    "mountain climbers": "left_knee",
    "plank":           "trunk",
    # fallback
    "unknown":         None,
}

# Joint groups for activity fingerprinting
JOINT_GROUPS = {
    "arms":      ["left_elbow", "right_elbow", "left_shoulder", "right_shoulder"],
    "legs":      ["left_knee",  "right_knee",  "left_hip",       "right_hip"],
    "core":      ["trunk"],
    "spread":    ["arm_spread", "leg_spread"],
}

HISTORY_LEN   = 900   # ~30 s at 30 fps
SMOOTH_WIN    = 7     # light smoothing — keeps latency low
FPS_ESTIMATE  = 30


# ─────────────────────────────────────────────────────────────────────────────
# Math helpers
# ─────────────────────────────────────────────────────────────────────────────
def _angle(a, b, c):
    """Angle in degrees at vertex b."""
    ba = a - b;  bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _smooth(arr, win=SMOOTH_WIN):
    """Gaussian-weighted smoothing — lower latency than Savitzky-Golay."""
    if len(arr) < 3:
        return arr.copy()
    w = min(win, len(arr))
    if w % 2 == 0:
        w -= 1
    w = max(3, w)
    kernel = np.exp(-0.5 * np.linspace(-2, 2, w) ** 2)
    kernel /= kernel.sum()
    return np.convolve(arr, kernel, mode="same")


def _zero_crossing_reps(signal, min_amplitude=3.0, min_half_period=5):
    """
    Count reps via zero-crossings of the velocity (first derivative).

    A rep = one full oscillation = two direction reversals.
    Ignores reversals smaller than min_amplitude (noise filtering).
    min_half_period: minimum frames between valid reversals.

    Returns (rep_count, list_of_reversal_frame_indices)
    """
    if len(signal) < 10:
        return 0, []

    s   = _smooth(np.array(signal, dtype=float))
    vel = np.diff(s)                    # velocity
    vel = _smooth(vel, win=5)           # smooth velocity too

    reversals = []
    last_rev  = -min_half_period - 1
    last_sign = np.sign(vel[0])

    for i in range(1, len(vel)):
        sign = np.sign(vel[i])
        if sign == 0:
            continue
        if sign != last_sign:
            # Check amplitude of the swing leading to this reversal
            start = max(0, last_rev)
            swing = float(np.max(s[start:i+1]) - np.min(s[start:i+1]))
            if swing >= min_amplitude and (i - last_rev) >= min_half_period:
                reversals.append(i)
                last_rev  = i
                last_sign = sign

    # Each full rep = 2 reversals (down + up, or up + down)
    rep_count = len(reversals) // 2
    return rep_count, reversals


# ─────────────────────────────────────────────────────────────────────────────
# Per-angle tracker
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class AngleTracker:
    name:   str
    values: deque = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))

    def push(self, v):
        self.values.append(float(v))

    def array(self):
        return np.array(self.values)

    def rep_count(self):
        a = self.array()
        if len(a) < 20:
            return 0, []
        # Amplitude filter: only count if angle actually swings meaningfully
        rng = float(a.max() - a.min())
        min_amp = max(8.0, rng * 0.15)   # at least 15% of total range
        return _zero_crossing_reps(a, min_amplitude=min_amp)

    def activity(self):
        """RMS velocity — how much is this joint moving right now (last 60 frames)."""
        a = self.array()
        if len(a) < 5:
            return 0.0
        recent = a[-60:]
        vel    = np.abs(np.diff(_smooth(recent)))
        return float(vel.mean()) if len(vel) else 0.0

    def stats(self):
        a = self.array()
        if len(a) < 5:
            return {}
        rng = float(a.max() - a.min())
        n_reps, _ = self.rep_count()
        return {
            "mean":   round(float(a.mean()), 1),
            "min":    round(float(a.min()),  1),
            "max":    round(float(a.max()),  1),
            "range":  round(rng, 1),
            "reps":   n_reps,
            "activity": round(self.activity(), 3),
        }

    def current(self):
        return self.values[-1] if self.values else None


# ─────────────────────────────────────────────────────────────────────────────
# Activity fingerprint  (used for change detection)
# ─────────────────────────────────────────────────────────────────────────────
def _activity_fingerprint(trackers):
    """
    Returns a normalised vector of per-group activity levels.
    Shape: (4,)  for [arms, legs, core, spread]
    """
    vec = []
    for group_name, angle_names in JOINT_GROUPS.items():
        acts = [trackers[n].activity() for n in angle_names if n in trackers]
        vec.append(float(np.mean(acts)) if acts else 0.0)
    vec = np.array(vec, dtype=float)
    norm = vec.sum() + 1e-8
    return vec / norm


def _cosine_distance(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(1.0 - np.dot(a, b) / denom)


# ─────────────────────────────────────────────────────────────────────────────
# Main analyzer
# ─────────────────────────────────────────────────────────────────────────────
class MotionAnalyzer:
    """
    Call .update(landmark_list) every frame.
    Call .rep_count(exercise)   -> (n_reps, signal_name, reversal_indices)
    Call .should_reclassify()   -> bool  (fingerprint changed significantly)
    Call .summarize()           -> str   (natural language for LLM)
    Call .get_graph_data(exercise) -> dict for plotting
    """

    def __init__(self):
        self.trackers    = {name: AngleTracker(name) for name, *_ in ANGLE_DEFS}
        self.frame_count = 0

        # Fingerprint history for change detection
        self._fp_history         = deque(maxlen=30)   # ~1 s of fingerprints at 30 fps
        self._last_fp_snapshot   = None               # fingerprint at last classification
        self._reclassify_flag    = False
        self._min_frames_between = 90                 # don't reclassify more than ~3s apart
        self._last_classify_frame = 0

    # ── ingest ────────────────────────────────────────────────────────────────
    def update(self, landmarks):
        if landmarks is None:
            return
        lm_list = landmarks.landmark if hasattr(landmarks, "landmark") else landmarks
        lm = np.array([[l.x, l.y, l.z] for l in lm_list])
        if len(lm) < 33:
            return

        self.frame_count += 1

        for name, a, b, c in ANGLE_DEFS:
            self.trackers[name].push(_angle(lm[a], lm[b], lm[c]))

        # Update fingerprint every 15 frames (~0.5 s)
        if self.frame_count % 15 == 0:
            fp = _activity_fingerprint(self.trackers)
            self._fp_history.append(fp)
            self._check_exercise_change()

    # ── change detection ──────────────────────────────────────────────────────
    def _check_exercise_change(self):
        if len(self._fp_history) < 4:
            return
        frames_since = self.frame_count - self._last_classify_frame
        if frames_since < self._min_frames_between:
            return

        current_fp = np.mean(list(self._fp_history)[-4:], axis=0)

        if self._last_fp_snapshot is None:
            # First time — take a snapshot but don't trigger yet
            self._last_fp_snapshot = current_fp
            return

        dist = _cosine_distance(current_fp, self._last_fp_snapshot)

        # Trigger if activity pattern shifted meaningfully
        if dist > 0.25:
            self._reclassify_flag    = True
            self._last_fp_snapshot   = current_fp
            self._last_classify_frame = self.frame_count

    def should_reclassify(self):
        """Returns True once, then resets flag."""
        if self._reclassify_flag and self.frame_count > 90:
            self._reclassify_flag = False
            return True
        return False

    def force_reclassify(self):
        """Called by user pressing C."""
        if self.frame_count > 30:
            self._reclassify_flag    = True
            self._last_fp_snapshot   = None   # force snapshot reset too

    # ── rep counting ──────────────────────────────────────────────────────────
    def rep_count(self, exercise=None):
        """
        Returns (n_reps, signal_name, reversal_indices).

        If exercise is known, use its designated signal.
        Otherwise, pick the angle with the highest rep count (most active).
        """
        # If we know the exercise, use its signal
        if exercise and exercise.lower() in EXERCISE_SIGNAL:
            sig_name = EXERCISE_SIGNAL[exercise.lower()]
            if sig_name and sig_name in self.trackers:
                n, revs = self.trackers[sig_name].rep_count()
                return n, sig_name, revs

        # Otherwise pick the tracker with the highest rep count
        best_n, best_name, best_revs = 0, "left_knee", []
        for name, tracker in self.trackers.items():
            n, revs = tracker.rep_count()
            if n > best_n:
                best_n, best_name, best_revs = n, name, revs

        return best_n, best_name, best_revs

    # ── graph data ────────────────────────────────────────────────────────────
    def get_graph_data(self, exercise=None):
        _, sig_name, reversals = self.rep_count(exercise)
        tracker = self.trackers[sig_name]
        arr     = tracker.array()
        s       = _smooth(arr) if len(arr) > 3 else arr
        return {
            "signal":    s,
            "reversals": np.array(reversals),
            "key":       sig_name,
        }

    # ── activity snapshot for display ─────────────────────────────────────────
    def activity_summary(self):
        """Returns dict of group_name -> activity level (0-1) for HUD bar."""
        total = {}
        for group, names in JOINT_GROUPS.items():
            acts = [self.trackers[n].activity() for n in names if n in self.trackers]
            total[group] = float(np.mean(acts)) if acts else 0.0
        max_v = max(total.values()) + 1e-8
        return {k: v / max_v for k, v in total.items()}

    # ── natural-language summary ───────────────────────────────────────────────
    def summarize(self):
        if self.frame_count < 30:
            return ""

        lines = []
        dur   = self.frame_count / FPS_ESTIMATE
        lines.append("Observation window: ~{:.0f} seconds ({} frames).".format(
            dur, self.frame_count))

        # Activity fingerprint description
        fp     = _activity_fingerprint(self.trackers)
        groups = list(JOINT_GROUPS.keys())
        dominant = groups[int(np.argmax(fp))]
        lines.append("Dominant joint group: {} (activity weights: {}).".format(
            dominant,
            ", ".join("{} {:.2f}".format(g, fp[i]) for i, g in enumerate(groups))
        ))

        # Per-joint stats, most active first
        stats_list = []
        for name, tracker in self.trackers.items():
            st = tracker.stats()
            if st and st["range"] > 5:
                stats_list.append((name, st))
        stats_list.sort(key=lambda x: -x[1]["range"])

        lines.append("\nJoint angle observations (range > 5 degrees):")
        for name, st in stats_list[:10]:
            lines.append(
                "  {}: avg {:.0f} deg, swings {:.0f}-{:.0f} deg "
                "(range {:.0f} deg, ~{} reps detected, activity {:.3f})".format(
                    name.replace("_", " "),
                    st["mean"], st["min"], st["max"],
                    st["range"], st["reps"], st["activity"]
                )
            )

        # High-level observations
        lines.append("\nKey observations:")
        arm_act  = (self.trackers["left_elbow"].activity() +
                    self.trackers["right_elbow"].activity()) / 2
        leg_act  = (self.trackers["left_knee"].activity() +
                    self.trackers["right_knee"].activity()) / 2
        core_act = self.trackers["trunk"].activity()
        spr_act  = (self.trackers["arm_spread"].activity() +
                    self.trackers["leg_spread"].activity()) / 2

        if arm_act > leg_act and arm_act > core_act:
            le = self.trackers["left_elbow"].stats()
            re = self.trackers["right_elbow"].stats()
            if le and re:
                lines.append("  - Arms are the primary movers (elbow range {:.0f} deg avg).".format(
                    (le["range"] + re["range"]) / 2))
        if leg_act > arm_act and leg_act > core_act:
            lk = self.trackers["left_knee"].stats()
            if lk:
                lines.append("  - Legs are the primary movers (knee range {:.0f} deg, min {:.0f} deg).".format(
                    lk["range"], lk["min"]))
        if core_act > arm_act and core_act > leg_act:
            tr = self.trackers["trunk"].stats()
            if tr:
                lines.append("  - Core/trunk is primary mover (trunk angle range {:.0f} deg).".format(
                    tr["range"]))
        if spr_act > 0.5 * max(arm_act, leg_act, core_act):
            asp = self.trackers["arm_spread"].stats()
            lsp = self.trackers["leg_spread"].stats()
            if asp and lsp:
                lines.append("  - Lateral spreading motion detected (arm spread {:.0f} deg, leg spread {:.0f} deg range).".format(
                    asp["range"], lsp["range"]))

        tr = self.trackers["trunk"].stats()
        if tr and tr["mean"] < 145:
            lines.append("  - Significant forward torso lean (avg trunk angle {:.0f} deg).".format(
                tr["mean"]))
        elif tr:
            lines.append("  - Torso relatively upright (avg trunk angle {:.0f} deg).".format(
                tr["mean"]))

        # Rep counts per signal
        lines.append("\nRep counts per joint signal:")
        rep_counts = []
        for name, tracker in self.trackers.items():
            n, _ = tracker.rep_count()
            if n > 0:
                rep_counts.append("  {}: {} reps".format(name.replace("_"," "), n))
        lines.extend(rep_counts if rep_counts else ["  (no clear repetitions detected yet)"])

        return "\n".join(lines)
