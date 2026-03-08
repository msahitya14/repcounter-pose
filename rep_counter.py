"""
rep_counter.py
==============
Real-time rep counter + form feedback using MediaPipe + trained classifier.

Usage:
  python rep_counter.py --source 0              # webcam
  python rep_counter.py --source video.mp4      # video file
  python rep_counter.py --source 0 --clf models/classifier.pkl --output out.mp4
"""

import argparse, collections, time
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from stage_labels import normalize_label, split_label

# ── MediaPipe landmark indices ────────────────────────────────────────────────
LM = {
    "nose":0,"left_shoulder":11,"right_shoulder":12,
    "left_elbow":13,"right_elbow":14,
    "left_wrist":15,"right_wrist":16,
    "left_hip":23,"right_hip":24,
    "left_knee":25,"right_knee":26,
    "left_ankle":27,"right_ankle":28,
}

MP_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),   # arms
    (11,23),(12,24),(23,24),                    # torso
    (23,25),(25,27),(24,26),(26,28),            # legs
    (0,11),(0,12),                              # head-shoulder
]

# ── Exercise state machines ───────────────────────────────────────────────────
# Maps base exercise name → (up_label, down_label, rep_signal_fn, form_checks)
EXERCISES = {
    "jumpingjack": {
        "up":   "jumpingjack_up",
        "down": "jumpingjack_down",
        "rest": "jumpingjack_rest",
        "signal_joint": "left_wrist",   # wrist Y for signal
        "invert": True,                 # up = low Y value
        "form": [
            ("arms_raised",    "Raise arms fully overhead"),
            ("legs_spread",    "Spread legs wider"),
        ]
    },
    "pushup": {
        "up":   "pushup_up",
        "down": "pushup_down",
        "rest": "pushup_rest",
        "signal_joint": "left_elbow",
        "invert": False,
        "form": [
            ("back_straight",  "Keep your back straight"),
            ("elbows_close",   "Keep elbows closer to body"),
            ("full_extension", "Fully extend arms at top"),
        ]
    },
    "pullup": {
        "up":   "pullup_up",
        "down": "pullup_down",
        "rest": "pullup_rest",
        "signal_joint": "left_wrist",
        "invert": True,
        "form": [
            ("chin_over_bar",  "Get chin above the bar"),
            ("full_extension", "Fully extend at bottom"),
            ("no_swinging",    "Control the swing"),
        ]
    },
    "situp": {
        "up":   "situp_up",
        "down": "situp_down",
        "rest": "situp_rest",
        "signal_joint": "nose",
        "invert": True,
        "form": [
            ("full_crunch",    "Crunch all the way up"),
            ("controlled",     "Lower slowly"),
            ("neck_neutral",   "Keep neck neutral"),
        ]
    },
    "squat": {
        "up":   "squat_up",
        "down": "squat_down",
        "rest": "squat_rest",
        "signal_joint": "left_knee",
        "invert": False,
        "form": [
            ("depth",          "Squat deeper – thighs parallel"),
            ("knees_out",      "Push knees outward"),
            ("back_upright",   "Keep torso upright"),
            ("weight_heels",   "Weight on your heels"),
        ]
    },
}

class StageSmoother:
    def __init__(self, window=5, min_votes=3):
        self.window = window
        self.min_votes = min_votes
        self.buf = collections.deque(maxlen=window)

    def reset(self):
        self.buf.clear()

    def update(self, stage: str) -> str:
        if stage not in {"up", "down", "rest"}:
            return "unknown"
        self.buf.append(stage)
        counts = collections.Counter(self.buf)
        top_stage, top_votes = counts.most_common(1)[0]
        return top_stage if top_votes >= self.min_votes else "unknown"


class ExerciseSmoother:
    def __init__(self, window=15, min_votes=8):
        self.buf = collections.deque(maxlen=window)
        self.min_votes = min_votes

    def reset(self):
        self.buf.clear()

    def update(self, exercise: str) -> str:
        if exercise not in EXERCISES:
            return "unknown"
        self.buf.append(exercise)
        counts = collections.Counter(self.buf)
        top_ex, top_votes = counts.most_common(1)[0]
        return top_ex if top_votes >= self.min_votes else "unknown"


class StageTransitionCounter:
    """
    Counts one rep for each stable down->up completion.
    """
    def __init__(self, min_down_frames=2, min_up_frames=2, cooldown_frames=8):
        self.min_down_frames = max(1, int(min_down_frames))
        self.min_up_frames = max(1, int(min_up_frames))
        self.cooldown_frames = max(0, int(cooldown_frames))
        self.reps = 0
        self._seen_down = False
        self._down_streak = 0
        self._up_streak = 0
        self._frame_idx = 0
        self._last_rep_frame = -10**9

    def reset(self):
        self.reps = 0
        self._seen_down = False
        self._down_streak = 0
        self._up_streak = 0
        self._frame_idx = 0
        self._last_rep_frame = -10**9

    def update(self, stage: str) -> int:
        self._frame_idx += 1
        if stage == "down":
            self._down_streak += 1
            self._up_streak = 0
            if self._down_streak >= self.min_down_frames:
                self._seen_down = True
        elif stage == "up":
            self._up_streak += 1
            self._down_streak = 0
            can_count = (
                self._seen_down
                and self._up_streak >= self.min_up_frames
                and (self._frame_idx - self._last_rep_frame) >= self.cooldown_frames
            )
            if can_count:
                self.reps += 1
                self._seen_down = False
                self._up_streak = 0
                self._last_rep_frame = self._frame_idx
        else:
            self._down_streak = 0
            self._up_streak = 0
        return self.reps


class SignalRepCounter:
    """
    Rep counter from motion cycles using adaptive low/high thresholds.
    Counts on high->low transition (down->up) of the selected joint signal.
    """
    def __init__(self, min_range=0.06, cooldown_frames=8):
        self.min_range = float(min_range)
        self.cooldown_frames = max(0, int(cooldown_frames))
        self.values = collections.deque(maxlen=120)
        self.reps = 0
        self.phase = "unknown"  # "high" or "low"
        self._frame_idx = 0
        self._last_rep_frame = -10**9

    def reset(self):
        self.values.clear()
        self.reps = 0
        self.phase = "unknown"
        self._frame_idx = 0
        self._last_rep_frame = -10**9

    def update(self, value: float) -> int:
        self._frame_idx += 1
        self.values.append(float(value))
        if len(self.values) < 20:
            return self.reps

        arr = np.array(self.values, dtype=float)
        lo = np.percentile(arr, 20)
        hi = np.percentile(arr, 80)
        rng = hi - lo
        if rng < self.min_range:
            return self.reps

        low_thr = lo + 0.25 * rng
        high_thr = lo + 0.75 * rng
        new_phase = self.phase
        if value <= low_thr:
            new_phase = "low"
        elif value >= high_thr:
            new_phase = "high"

        if self.phase == "high" and new_phase == "low":
            if (self._frame_idx - self._last_rep_frame) >= self.cooldown_frames:
                self.reps += 1
                self._last_rep_frame = self._frame_idx

        self.phase = new_phase
        return self.reps


class SetTracker:
    """
    Tracks per-set reps from a monotonic total rep count.
    """
    def __init__(self, reps_per_set=10, target_sets=3):
        self.reps_per_set = max(1, int(reps_per_set))
        self.target_sets = max(1, int(target_sets))
        self.total_reps = 0
        self.current_set = 1
        self.reps_in_set = 0
        self.completed_sets = 0
        self.workout_done = False

    def reset(self):
        self.total_reps = 0
        self.current_set = 1
        self.reps_in_set = 0
        self.completed_sets = 0
        self.workout_done = False

    def update_total_reps(self, total_reps: int) -> bool:
        """
        Returns True when a new rep event is consumed.
        """
        if total_reps <= self.total_reps or self.workout_done:
            return False

        rep_delta = total_reps - self.total_reps
        self.total_reps = total_reps

        for _ in range(rep_delta):
            self.reps_in_set += 1
            if self.reps_in_set >= self.reps_per_set:
                self.completed_sets += 1
                self.reps_in_set = 0
                if self.completed_sets >= self.target_sets:
                    self.workout_done = True
                    self.current_set = self.target_sets
                else:
                    self.current_set = self.completed_sets + 1
        return True


# ── Angle helpers ──────────────────────────────────────────────────────────────
def angle3(a, b, c):
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def extract_mediapipe_features(landmarks_xyz: np.ndarray) -> dict:
    """
    landmarks_xyz: (33, 3) normalized MediaPipe landmarks
    Returns feature dict matching the training feat_cols.
    In live mode we only have landmarks; build derived features on-the-fly.
    """
    lm = landmarks_xyz
    feats = {}
    # Raw landmark coords
    MP_NAMES = [
        "nose","left_eye_inner","left_eye","left_eye_outer",
        "right_eye_inner","right_eye","right_eye_outer",
        "left_ear","right_ear","mouth_left","mouth_right",
        "left_shoulder","right_shoulder","left_elbow","right_elbow",
        "left_wrist","right_wrist","left_pinky_1","right_pinky_1",
        "left_index_1","right_index_1","left_thumb_2","right_thumb_2",
        "left_hip","right_hip","left_knee","right_knee",
        "left_ankle","right_ankle","left_heel","right_heel",
        "left_foot_index","right_foot_index",
    ]
    for i, name in enumerate(MP_NAMES):
        if i < len(lm):
            feats[f"landmarks__x_{name}"] = float(lm[i, 0])
            feats[f"landmarks__y_{name}"] = float(lm[i, 1])
            feats[f"landmarks__z_{name}"] = float(lm[i, 2])

    # Key angles
    angle_defs = {
        "left_elbow":  (11, 13, 15),
        "right_elbow": (12, 14, 16),
        "left_knee":   (23, 25, 27),
        "right_knee":  (24, 26, 28),
        "left_hip":    (11, 23, 25),
        "right_hip":   (12, 24, 26),
        "left_shoulder":(13,11,23),
        "right_shoulder":(14,12,24),
    }
    for aname, (a, b, c) in angle_defs.items():
        if max(a, b, c) < len(lm):
            feats[f"angles__{aname}_angle"] = angle3(lm[a], lm[b], lm[c])

    # Key 3D distances
    dist_pairs = [
        ("wrists",     15, 16),
        ("ankles",     27, 28),
        ("shoulders",  11, 12),
        ("hips",       23, 24),
        ("l_wrist_hip",15, 23),
        ("r_wrist_hip",16, 24),
        ("l_hand_head",15, 0),
        ("r_hand_head",16, 0),
    ]
    for dname, i, j in dist_pairs:
        if max(i, j) < len(lm):
            feats[f"dist3d__{dname}"] = float(np.linalg.norm(lm[i] - lm[j]))

    return feats


# ── Form feedback ──────────────────────────────────────────────────────────────
def check_form(exercise: str, lm: np.ndarray) -> list[tuple[str, bool]]:
    """
    Returns list of (feedback_text, is_good) tuples.
    lm: (33, 3)
    """
    if len(lm) < 29 or exercise not in EXERCISES:
        return []

    issues = []

    def ang(a, b, c):
        return angle3(lm[a], lm[b], lm[c])

    if exercise == "squat":
        knee_angle   = ang(23, 25, 27)   # left hip→knee→ankle
        torso_angle  = ang(25, 23, 11)   # knee→hip→shoulder (lean)
        knee_dist    = abs(lm[25, 0] - lm[27, 0])  # knee vs ankle X
        issues = [
            ("Squat deeper – thighs parallel",  knee_angle > 100),
            ("Keep torso upright",              torso_angle < 60),
            ("Push knees over toes",            knee_dist < 0.03),
        ]

    elif exercise == "pushup":
        elbow_angle  = ang(11, 13, 15)
        hip_y        = (lm[23, 1] + lm[24, 1]) / 2
        shoulder_y   = (lm[11, 1] + lm[12, 1]) / 2
        sag          = hip_y - shoulder_y
        issues = [
            ("Fully extend arms at top",   elbow_angle < 150),
            ("Keep back straight – no sag", sag > 0.06),
            ("Lower chest to ground",      elbow_angle > 80),
        ]

    elif exercise == "pullup":
        elbow_angle  = ang(11, 13, 15)
        wrist_y      = (lm[15, 1] + lm[16, 1]) / 2
        shoulder_y   = (lm[11, 1] + lm[12, 1]) / 2
        issues = [
            ("Pull higher – chin over bar", wrist_y > shoulder_y - 0.05),
            ("Fully extend at bottom",      elbow_angle < 150),
        ]

    elif exercise == "situp":
        torso_angle  = ang(25, 23, 11)
        issues = [
            ("Crunch all the way up",      torso_angle < 50),
            ("Keep neck neutral",          True),  # always remind
        ]

    elif exercise == "jumpingjack":
        wrist_dist   = abs(lm[15, 0] - lm[16, 0])
        ankle_dist   = abs(lm[27, 0] - lm[28, 0])
        wrist_y_avg  = (lm[15, 1] + lm[16, 1]) / 2
        shoulder_y   = (lm[11, 1] + lm[12, 1]) / 2
        issues = [
            ("Raise arms fully overhead",  wrist_y_avg > shoulder_y - 0.05),
            ("Spread legs wider",          ankle_dist < 0.25),
        ]

    return [(text, not bad) for text, bad in issues]


# ── Signal smoothing + rep counting ───────────────────────────────────────────
def smooth(s, w=9):
    if len(s) < w:
        return np.array(s, dtype=float)
    w = w if w % 2 == 1 else w + 1
    w = min(w, len(s) if len(s) % 2 == 1 else len(s) - 1)
    return savgol_filter(s, window_length=max(3, w), polyorder=2)


def count_reps_stateful(signal, invert, prom_frac=0.15):
    s = smooth(np.array(signal, dtype=float))
    if invert:
        s = -s
    rng = s.max() - s.min()
    if rng < 1e-3:
        return 0, np.array([])
    peaks, _ = find_peaks(s, prominence=rng * prom_frac,
                          distance=max(3, len(s) // 20))
    return len(peaks), peaks


# ── Graph renderer ─────────────────────────────────────────────────────────────
def render_graph(signal, peaks, exercise, n_reps, w=640, h=150):
    dpi = 80
    fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
    fig.patch.set_facecolor("#080c14")
    ax.set_facecolor("#080c14")

    if len(signal) > 1:
        t = np.arange(len(signal))
        s = smooth(signal)
        ax.plot(t, s, color="#00ffe0", lw=1.8, alpha=0.95)
        ax.fill_between(t, s.min(), s, alpha=0.12, color="#00ffe0")
        if len(peaks):
            ax.scatter(peaks, s[peaks], color="#ff3c6e", s=50, zorder=5, marker="v")

    ax.set_xlim(0, max(len(signal) - 1, 1))
    ax.tick_params(colors="#3a4a6a", labelsize=6)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1a2540")
    ax.set_title(
        f"{'  '.join(exercise.upper().split('_'))}   ·   {n_reps} REPS",
        color="#e0eaff", fontsize=8, pad=2, fontfamily="monospace"
    )
    fig.tight_layout(pad=0.2)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# ── HUD overlay ───────────────────────────────────────────────────────────────
COLORS = {
    "good":  (0, 230, 120),
    "bad":   (0, 100, 255),
    "info":  (0, 220, 255),
    "dim":   (60, 80, 120),
    "white": (230, 240, 255),
    "rep":   (0, 255, 200),
}

def draw_hud(frame, exercise, stage, n_reps, confidence, pose_label,
             form_feedback, fps, current_set, reps_in_set, reps_per_set, workout_done):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 64), (5, 8, 18), -1)
    cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)

    # Rep counter (big)
    cv2.putText(frame, f"{n_reps}", (16, 54),
                cv2.FONT_HERSHEY_DUPLEX, 2.0, COLORS["rep"], 3, cv2.LINE_AA)
    cv2.putText(frame, "REPS", (76, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS["dim"], 1, cv2.LINE_AA)
    cv2.putText(frame, f"SET {current_set}  {reps_in_set}/{reps_per_set}",
                (16, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS["white"], 1, cv2.LINE_AA)

    # Exercise + confidence
    ex_str = exercise.upper().replace("_", " ")
    cv2.putText(frame, f"{ex_str}  {confidence*100:.0f}%",
                (140, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.72, COLORS["info"], 1, cv2.LINE_AA)

    # Pose label
    cv2.putText(frame, pose_label.replace("_", " "),
                (140, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["dim"], 1, cv2.LINE_AA)
    cv2.putText(frame, f"stage: {stage}",
                (w - 160, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["dim"], 1, cv2.LINE_AA)
    if workout_done:
        cv2.putText(frame, "WORKOUT COMPLETE", (w // 2 - 120, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS["good"], 2, cv2.LINE_AA)

    # FPS
    cv2.putText(frame, f"{fps:.0f} fps", (w - 90, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["dim"], 1, cv2.LINE_AA)

    # Form feedback panel (right side)
    if form_feedback:
        panel_x = w - 300
        for i, (text, is_good) in enumerate(form_feedback[:4]):
            cy = 95 + i * 26
            color = COLORS["good"] if is_good else COLORS["bad"]
            icon  = "✓" if is_good else "!"
            cv2.putText(frame, f"{icon} {text}", (panel_x, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)

    return frame


def draw_skeleton(frame, results, conf_thr=0.5):
    try:
        import mediapipe as mp
        lms = results.pose_landmarks
        if lms is None:
            return frame, None
        h, w = frame.shape[:2]
        pts = np.array([[l.x * w, l.y * h, l.z] for l in lms.landmark])
        vis = np.array([l.visibility for l in lms.landmark])
        for a, b in MP_CONNECTIONS:
            if vis[a] > conf_thr and vis[b] > conf_thr:
                cv2.line(frame,
                         tuple(pts[a, :2].astype(int)),
                         tuple(pts[b, :2].astype(int)),
                         (255, 200, 0), 2, cv2.LINE_AA)
        for i in range(len(pts)):
            if vis[i] > conf_thr:
                cv2.circle(frame, tuple(pts[i, :2].astype(int)),
                           4, (0, 255, 128), -1, cv2.LINE_AA)
        # Return normalized coords for feature extraction
        norm = np.array([[l.x, l.y, l.z] for l in lms.landmark])
        return frame, norm
    except Exception:
        return frame, None


# ── Main loop ─────────────────────────────────────────────────────────────────
def run(source, clf_path, output_path, min_confidence, min_stage_frames, rep_cooldown_frames,
        reps_per_set, target_sets):
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
    except ImportError:
        raise ImportError("Run: pip install mediapipe")

    clf_dict = None
    if clf_path and Path(clf_path).exists():
        import pickle
        with open(clf_path, "rb") as f:
            clf_dict = pickle.load(f)
        from train_classifier import predict_pose
        print(f"✅ Classifier loaded: {clf_path}")
    else:
        print("⚠  No classifier – pose labels shown as 'unknown'")

    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {source}")

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_h = fh + 150
    writer = None
    if output_path:
        fps_out = cap.get(cv2.CAP_PROP_FPS) or 30
        writer  = cv2.VideoWriter(output_path,
                                  cv2.VideoWriter_fourcc(*"mp4v"),
                                  fps_out, (fw, total_h))

    signal_buf     = collections.deque(maxlen=400)
    exercise       = "squat"
    stage          = "rest"
    pose_label     = "unknown"
    confidence     = 0.0
    form_feedback  = []
    stage_smoother = StageSmoother(window=5, min_votes=3)
    exercise_smoother = ExerciseSmoother(window=15, min_votes=8)
    rep_counter    = StageTransitionCounter(
        min_down_frames=min_stage_frames,
        min_up_frames=min_stage_frames,
        cooldown_frames=rep_cooldown_frames,
    )
    signal_counter = SignalRepCounter(min_range=0.06, cooldown_frames=rep_cooldown_frames)
    set_tracker    = SetTracker(reps_per_set=reps_per_set, target_sets=target_sets)
    frame_idx      = 0
    t0             = time.time()
    ex_cfg         = EXERCISES[exercise]

    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            frame, lm_norm = draw_skeleton(frame, result)

            fps = frame_idx / max(time.time() - t0, 0.01)
            frame_idx += 1

            if lm_norm is not None:
                # Classify every frame
                if clf_dict is not None:
                    from train_classifier import predict_pose
                    feats    = extract_mediapipe_features(lm_norm)
                    raw_label, confidence, _ = predict_pose(clf_dict, feats)
                    pose_label = normalize_label(raw_label)
                    pred_ex, pred_stage = split_label(pose_label)
                    if confidence < min_confidence:
                        pred_ex = "unknown"
                        pred_stage = "unknown"
                    stable_ex = exercise_smoother.update(pred_ex)
                    if stable_ex in EXERCISES:
                        if stable_ex != exercise:
                            exercise = stable_ex
                            pose_label = f"{exercise}_{stage}"
                            confidence = 0.0
                            pred_stage = "unknown"
                            signal_buf.clear()
                            stage_smoother.reset()
                            rep_counter.reset()
                            signal_counter.reset()
                            set_tracker.reset()
                        observed_stage = stage_smoother.update(pred_stage)
                        if observed_stage != "unknown":
                            stage = observed_stage
                        # Only advance counter on frames where stage is stable.
                        if observed_stage in {"up", "down", "rest"}:
                            rep_counter.update(observed_stage)
                        else:
                            rep_counter.update("unknown")
                    elif pred_ex in EXERCISES:
                        # Keep collecting exercise votes, but don't switch yet.
                        pass
                    else:
                        stage = "unknown"

                # Build signal
                ex_cfg = EXERCISES.get(exercise, EXERCISES["squat"])
                jname  = ex_cfg["signal_joint"]
                jidx   = LM.get(jname, 25)
                if jidx < len(lm_norm):
                    sig_val = float(lm_norm[jidx, 1])
                    signal_buf.append(sig_val)
                    signal_counter.update(sig_val)

                # Form check every 5 frames
                if frame_idx % 5 == 0:
                    form_feedback = check_form(exercise, lm_norm)

            sig    = np.array(signal_buf)
            ex_cfg = EXERCISES.get(exercise, EXERCISES["squat"])
            n_reps_signal, peaks = count_reps_stateful(sig, ex_cfg["invert"])
            if clf_dict is not None:
                n_reps = max(rep_counter.reps, signal_counter.reps)
            else:
                n_reps = n_reps_signal
            set_tracker.update_total_reps(n_reps)

            draw_hud(
                frame, exercise, stage, n_reps, confidence,
                pose_label, form_feedback, fps,
                current_set=set_tracker.current_set,
                reps_in_set=set_tracker.reps_in_set,
                reps_per_set=set_tracker.reps_per_set,
                workout_done=set_tracker.workout_done,
            )

            graph = render_graph(sig, peaks, exercise, n_reps, w=fw, h=150)
            if graph.shape[1] != fw:
                graph = cv2.resize(graph, (fw, 150))
            combined = np.vstack([frame, graph])

            cv2.imshow("RepCounter", combined)
            if writer:
                writer.write(combined)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # Save final graph
    sig = np.array(signal_buf)
    n_reps_signal, peaks = count_reps_stateful(sig, ex_cfg["invert"])
    n_reps = rep_counter.reps if clf_dict is not None else n_reps_signal
    _save_final_graph(sig, peaks, exercise, n_reps)
    print(f"\n✅ Finished. Last exercise: {exercise} | Reps: {n_reps}")


def _save_final_graph(signal, peaks, exercise, n_reps, path="movement_graph.png"):
    fig, ax = plt.subplots(figsize=(13, 4))
    fig.patch.set_facecolor("#080c14"); ax.set_facecolor("#080c14")
    if len(signal) > 1:
        t = np.arange(len(signal))
        s = smooth(signal)
        ax.plot(t, s, color="#00ffe0", lw=2)
        ax.fill_between(t, s.min(), s, alpha=0.18, color="#00ffe0")
        if len(peaks):
            ax.scatter(peaks, s[peaks], color="#ff3c6e", s=90, zorder=5,
                       marker="v", label=f"{n_reps} reps")
            ax.legend(facecolor="#111828", labelcolor="white")
    ax.set_xlabel("Frame", color="#6070a0")
    ax.set_ylabel("Joint Y position", color="#6070a0")
    ax.set_title(
        f"Movement Graph – {exercise.upper().replace('_',' ')}  ·  {n_reps} REPS",
        color="#e0eaff", fontsize=13
    )
    ax.tick_params(colors="#3a4a6a")
    for sp in ax.spines.values():
        sp.set_edgecolor("#1a2540")
    plt.tight_layout()
    plt.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"📈 Graph saved → {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source",  required=True, help="Webcam index or video path")
    p.add_argument("--clf",     default=None,  help="Classifier .pkl path")
    p.add_argument("--output",  default=None,  help="Save output video")
    p.add_argument("--min_confidence", type=float, default=0.55,
                   help="Minimum classifier confidence for stage updates")
    p.add_argument("--min_stage_frames", type=int, default=2,
                   help="Minimum consecutive frames per stage before transitions are accepted")
    p.add_argument("--rep_cooldown_frames", type=int, default=8,
                   help="Minimum frame gap between counted reps")
    p.add_argument("--reps_per_set", type=int, default=10,
                   help="Number of reps needed to complete one set")
    p.add_argument("--target_sets", type=int, default=3,
                   help="Workout target number of sets")
    args = p.parse_args()
    run(
        args.source,
        args.clf,
        args.output,
        min_confidence=args.min_confidence,
        min_stage_frames=args.min_stage_frames,
        rep_cooldown_frames=args.rep_cooldown_frames,
        reps_per_set=args.reps_per_set,
        target_sets=args.target_sets,
    )
