"""
motion_analyzer.py
==================
Tracks joint angles from MediaPipe landmarks.
Counts reps via zero-crossing of the velocity signal (direction reversals).
Detects exercise changes via joint-activity fingerprint cosine distance.
Produces natural-language summaries for Ollama classification.
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe BlazePose 33-point indices
# ─────────────────────────────────────────────────────────────────────────────
LM = dict(
    nose=0,
    left_shoulder=11,  right_shoulder=12,
    left_elbow=13,     right_elbow=14,
    left_wrist=15,     right_wrist=16,
    left_hip=23,       right_hip=24,
    left_knee=25,      right_knee=26,
    left_ankle=27,     right_ankle=28,
)

# ─────────────────────────────────────────────────────────────────────────────
# Joint angles to track  (name, idx_a, idx_vertex, idx_c)
# NOTE: 'neck' intentionally excluded — too noisy from head wobble
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
    ("trunk",           25, 23, 11),   # knee->hip->shoulder (forward lean)
    ("arm_spread",      15, 11, 16),   # wrist->shoulder->wrist (lateral arm raise)
    ("leg_spread",      27, 23, 28),   # ankle->hip->ankle (legs apart)
]

# ─────────────────────────────────────────────────────────────────────────────
# Per-exercise best rep signal (updated by LLM at runtime)
# ─────────────────────────────────────────────────────────────────────────────
EXERCISE_SIGNAL = {
    "squats":           "left_knee",
    "lunges":           "left_knee",
    "push-ups":         "left_elbow",
    "pull-ups":         "left_elbow",
    "bicep curls":      "left_elbow",
    "shoulder press":   "left_shoulder",
    "sit-ups":          "trunk",
    "crunches":         "trunk",
    "jumping jacks":    "arm_spread",
    "burpees":          "left_knee",
    "mountain climbers":"left_knee",
    "plank":            "trunk",
    "unknown":          None,
}

# Minimum angle swing (degrees) required to count a reversal as real.
# Per-joint — prevents noise on joints that don't move much for a given exercise.
MIN_SWING = {
    "left_elbow":    15.0,
    "right_elbow":   15.0,
    "left_knee":     12.0,
    "right_knee":    12.0,
    "left_hip":      15.0,
    "right_hip":     15.0,
    "left_shoulder": 12.0,
    "right_shoulder":12.0,
    "trunk":         10.0,
    "arm_spread":    18.0,   # needs bigger swing to avoid arm drift noise
    "leg_spread":    12.0,
}

# Joint groups for activity fingerprinting
JOINT_GROUPS = {
    "arms":   ["left_elbow",  "right_elbow",  "left_shoulder",  "right_shoulder"],
    "legs":   ["left_knee",   "right_knee",   "left_hip",        "right_hip"],
    "core":   ["trunk"],
    "spread": ["arm_spread",  "leg_spread"],
}

HISTORY_LEN  = 900    # ~30 s at 30 fps
SMOOTH_WIN   = 9      # Gaussian smoothing window (frames)
FPS_ESTIMATE = 30
MIN_HALF_PERIOD = 8   # minimum frames between reversals (~4 fps max rep rate)


# ─────────────────────────────────────────────────────────────────────────────
# Math helpers
# ─────────────────────────────────────────────────────────────────────────────
def _angle(a, b, c):
    ba = a - b;  bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _smooth(arr, win=SMOOTH_WIN):
    """Gaussian-kernel smoothing."""
    if len(arr) < 3:
        return arr.copy()
    w = min(win, len(arr))
    if w % 2 == 0: w -= 1
    w = max(3, w)
    kernel = np.exp(-0.5 * np.linspace(-2, 2, w) ** 2)
    kernel /= kernel.sum()
    return np.convolve(arr, kernel, mode="same")


def _zero_crossing_reps(signal, min_amplitude=12.0, min_half_period=MIN_HALF_PERIOD):
    """
    Count reps as pairs of direction reversals in the joint angle velocity.

    A reversal is only counted if:
      - The velocity sign changes
      - The amplitude of the preceding swing >= min_amplitude degrees
      - At least min_half_period frames have passed since the last reversal

    Returns (rep_count, reversal_frame_indices_list)
    """
    if len(signal) < max(20, min_half_period * 3):
        return 0, []

    s   = _smooth(np.array(signal, dtype=float))
    vel = _smooth(np.diff(s), win=5)

    reversals = []
    last_rev  = -min_half_period - 1
    last_sign = 0

    # Find first non-zero sign
    for v in vel:
        if v != 0:
            last_sign = int(np.sign(v))
            break

    for i, v in enumerate(vel):
        if v == 0:
            continue
        sign = int(np.sign(v))
        if sign == last_sign:
            continue

        # Sign changed — check swing amplitude over the half-period
        seg_start = max(0, last_rev)
        seg       = s[seg_start : i + 1]
        swing     = float(seg.max() - seg.min()) if len(seg) > 1 else 0.0

        gap = i - last_rev
        if swing >= min_amplitude and gap >= min_half_period:
            reversals.append(i)
            last_rev  = i
        last_sign = sign

    rep_count = len(reversals) // 2
    return rep_count, reversals


# ─────────────────────────────────────────────────────────────────────────────
# Activity fingerprint helpers
# ─────────────────────────────────────────────────────────────────────────────
def _activity_fingerprint(trackers):
    vec = []
    for names in JOINT_GROUPS.values():
        acts = [trackers[n].activity() for n in names if n in trackers]
        vec.append(float(np.mean(acts)) if acts else 0.0)
    vec = np.array(vec, dtype=float)
    return vec / (vec.sum() + 1e-8)


def _cosine_distance(a, b):
    return float(1.0 - np.dot(a, b) /
                 (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ─────────────────────────────────────────────────────────────────────────────
# Per-joint angle tracker
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
        if len(a) < 30:
            return 0, []
        min_amp = MIN_SWING.get(self.name, 12.0)
        # Also require the joint to actually move meaningfully overall
        total_range = float(a.max() - a.min())
        if total_range < min_amp * 1.5:
            return 0, []
        return _zero_crossing_reps(a, min_amplitude=min_amp)

    def activity(self):
        """Mean absolute velocity over the last 2 seconds (~60 frames)."""
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
        n_reps, _ = self.rep_count()
        return {
            "mean":     round(float(a.mean()), 1),
            "min":      round(float(a.min()),  1),
            "max":      round(float(a.max()),  1),
            "range":    round(float(a.max() - a.min()), 1),
            "reps":     n_reps,
            "activity": round(self.activity(), 3),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Main analyzer
# ─────────────────────────────────────────────────────────────────────────────
class MotionAnalyzer:
    """
    update(landmark_list)            — feed one frame
    rep_count(exercise)              — (n, signal_name, reversal_indices)
    get_graph_data(exercise)         — dict for plotting
    should_reclassify()              — True when activity pattern shifts
    force_reclassify()               — called by user pressing C
    activity_summary()               — dict for HUD bars
    summarize()                      — natural language for LLM
    """

    def __init__(self):
        self.trackers    = {name: AngleTracker(name) for name, *_ in ANGLE_DEFS}
        self.frame_count = 0

        self._fp_history          = deque(maxlen=30)
        self._last_fp_snapshot    = None
        self._reclassify_flag     = False
        self._min_frames_between  = 120   # at least 4 s between reclassifications
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
        if self.frame_count % 15 == 0:
            fp = _activity_fingerprint(self.trackers)
            self._fp_history.append(fp)
            self._check_change()

    # ── change detection ──────────────────────────────────────────────────────
    def _check_change(self):
        if len(self._fp_history) < 6:
            return
        if self.frame_count - self._last_classify_frame < self._min_frames_between:
            return
        # Average last 6 snapshots (~3 s) vs snapshot at last classification
        current = np.mean(list(self._fp_history)[-6:], axis=0)
        if self._last_fp_snapshot is None:
            self._last_fp_snapshot = current
            return
        dist = _cosine_distance(current, self._last_fp_snapshot)
        if dist > 0.20:   # 20% shift in activity pattern
            self._reclassify_flag     = True
            self._last_fp_snapshot    = current
            self._last_classify_frame = self.frame_count

    def should_reclassify(self):
        if self._reclassify_flag and self.frame_count > 90:
            self._reclassify_flag = False
            return True
        return False

    def force_reclassify(self):
        if self.frame_count > 30:
            self._reclassify_flag     = True
            self._last_fp_snapshot    = None
            self._last_classify_frame = 0

    # ── rep counting ──────────────────────────────────────────────────────────
    def rep_count(self, exercise=None):
        """Returns (n_reps, signal_name, reversal_indices)."""
        # If exercise known, use designated signal
        if exercise:
            sig_name = EXERCISE_SIGNAL.get(exercise.lower())
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
        arr = self.trackers[sig_name].array()
        s   = _smooth(arr) if len(arr) > 3 else arr
        return {"signal": s, "reversals": np.array(reversals), "key": sig_name}

    # ── activity bars ─────────────────────────────────────────────────────────
    def activity_summary(self):
        totals = {}
        for group, names in JOINT_GROUPS.items():
            acts = [self.trackers[n].activity() for n in names if n in self.trackers]
            totals[group] = float(np.mean(acts)) if acts else 0.0
        max_v = max(totals.values()) + 1e-8
        return {k: v / max_v for k, v in totals.items()}

    # ── natural-language summary ───────────────────────────────────────────────
    def summarize(self):
        if self.frame_count < 30:
            return ""
        lines = []
        dur = self.frame_count / FPS_ESTIMATE
        lines.append("Observation window: ~{:.0f}s ({} frames).".format(
            dur, self.frame_count))

        fp     = _activity_fingerprint(self.trackers)
        groups = list(JOINT_GROUPS.keys())
        dom    = groups[int(np.argmax(fp))]
        lines.append("Most active joint group: {} | weights: {}.".format(
            dom, ", ".join("{} {:.2f}".format(g, fp[i]) for i, g in enumerate(groups))))

        # Joints with meaningful movement, sorted by range
        active = [(n, t.stats()) for n, t in self.trackers.items()]
        active = [(n, s) for n, s in active if s and s["range"] > 8]
        active.sort(key=lambda x: -x[1]["range"])

        lines.append("\nJoint movements (range > 8 deg, most active first):")
        for name, st in active[:9]:
            lines.append(
                "  {}: avg {:.0f}deg, range {:.0f}-{:.0f}deg "
                "(swing {:.0f}deg, {} reps, activity {:.2f})".format(
                    name.replace("_"," "), st["mean"],
                    st["min"], st["max"], st["range"],
                    st["reps"], st["activity"]))

        lines.append("\nKey observations:")
        arm_a  = np.mean([self.trackers[n].activity()
                          for n in ["left_elbow","right_elbow"]])
        leg_a  = np.mean([self.trackers[n].activity()
                          for n in ["left_knee","right_knee"]])
        core_a = self.trackers["trunk"].activity()
        spr_a  = np.mean([self.trackers[n].activity()
                          for n in ["arm_spread","leg_spread"]])

        if arm_a >= max(leg_a, core_a, spr_a):
            le = self.trackers["left_elbow"].stats()
            re = self.trackers["right_elbow"].stats()
            if le and re:
                avg_rng = (le["range"] + re["range"]) / 2
                avg_min = (le["min"]   + re["min"])   / 2
                lines.append("  - ARMS dominant: elbows swing {:.0f}deg avg, "
                              "min angle {:.0f}deg.".format(avg_rng, avg_min))
        if leg_a >= max(arm_a, core_a, spr_a):
            lk = self.trackers["left_knee"].stats()
            if lk:
                lines.append("  - LEGS dominant: knee swing {:.0f}deg, "
                              "min {:.0f}deg.".format(lk["range"], lk["min"]))
        if core_a >= max(arm_a, leg_a, spr_a):
            tr = self.trackers["trunk"].stats()
            if tr:
                lines.append("  - CORE dominant: trunk angle swing {:.0f}deg, "
                              "avg {:.0f}deg.".format(tr["range"], tr["mean"]))
        if spr_a > 0.4 * max(arm_a, leg_a, core_a):
            asp = self.trackers["arm_spread"].stats()
            lsp = self.trackers["leg_spread"].stats()
            if asp and lsp:
                lines.append("  - LATERAL motion: arm spread {:.0f}deg range, "
                              "leg spread {:.0f}deg range.".format(
                                  asp["range"], lsp["range"]))

        tr = self.trackers["trunk"].stats()
        if tr:
            lean = "upright" if tr["mean"] > 155 else "leaning forward"
            lines.append("  - Torso {}: avg trunk angle {:.0f}deg.".format(
                lean, tr["mean"]))

        lines.append("\nRep counts (reliable signals only):")
        counts = []
        for name, tracker in self.trackers.items():
            n, _ = tracker.rep_count()
            if n > 0:
                counts.append("  {}: {} reps".format(name.replace("_"," "), n))
        lines.extend(counts if counts else ["  (no clear reps yet)"])

        return "\n".join(lines)
