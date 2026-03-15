"""
rep_counter.py
==============
Live rep counter with synchronized:
  - ML stage classification (classifier.pkl from train_classifier.py)
  - Signal-based motion analysis / graphing
  - Optional LLM form-feedback enrichment

  MediaPipe Pose -> feature extraction -> stage classifier
                 -> stage-transition rep counting
                 -> MotionAnalyzer graph/activity
                 -> optional LLM on activity changes

Usage:
  python rep_counter.py --source 0 --clf models/classifier.pkl
  python rep_counter.py --source workout.mp4 --clf models/classifier.pkl --output out.mp4
  python rep_counter.py --source 0 --clf models/classifier.pkl --no_llm

Controls:  Q quit | R reset | C classify now | S save graph
"""

import argparse, collections, os, time, threading, urllib.request
import cv2, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from motion_analyzer import MotionAnalyzer, _smooth
from llm_classifier  import LLMClassifier, ClassificationResult
from stage_labels import normalize_label, split_label
from train_classifier import MP_LANDMARKS, load_model, predict_pose

# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe model download
# ─────────────────────────────────────────────────────────────────────────────
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/"   # full model = less lag
    "pose_landmarker_full.task"
)
MODEL_PATH = "pose_landmarker_full.task"

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading MediaPipe FULL pose model (~30 MB, better accuracy)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("  saved -> {}".format(MODEL_PATH))

# ─────────────────────────────────────────────────────────────────────────────
# Skeleton drawing
# ─────────────────────────────────────────────────────────────────────────────
MP_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(24,26),(26,28),
    (0,11),(0,12),
]

def draw_skeleton(frame, landmarks, conf_thr=0.35):
    if not landmarks:
        return
    h, w = frame.shape[:2]
    def vis(l): return getattr(l, "presence", getattr(l, "visibility", 1.0))
    pts = np.array([[l.x*w, l.y*h] for l in landmarks])
    v   = np.array([vis(l) for l in landmarks])
    for a, b in MP_CONNECTIONS:
        if a < len(pts) and b < len(pts) and v[a] > conf_thr and v[b] > conf_thr:
            cv2.line(frame, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)),
                     (255,200,0), 2, cv2.LINE_AA)
    for i in range(len(pts)):
        if v[i] > conf_thr:
            cv2.circle(frame, tuple(pts[i].astype(int)), 4, (0,255,128), -1, cv2.LINE_AA)

# ─────────────────────────────────────────────────────────────────────────────
# Figure -> BGR helper (matplotlib 3.8+ compatible)
# ─────────────────────────────────────────────────────────────────────────────
def _fig_to_bgr(fig):
    fig.canvas.draw()
    try:
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        return cv2.cvtColor(buf.reshape(h, w, 4), cv2.COLOR_RGBA2BGR)
    except AttributeError:
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        return cv2.cvtColor(buf.reshape(h, w, 3), cv2.COLOR_RGB2BGR)

# ─────────────────────────────────────────────────────────────────────────────
# Movement graph  — shows angle signal + reversal markers
# ─────────────────────────────────────────────────────────────────────────────
def render_graph(graph_data, exercise, n_reps, width, height=160):
    dpi = 80
    fig, ax = plt.subplots(figsize=(width/dpi, height/dpi), dpi=dpi)
    fig.patch.set_facecolor("#07090f")
    ax.set_facecolor("#07090f")

    sig  = graph_data["signal"]
    revs = graph_data["reversals"]
    key  = graph_data["key"].replace("_", " ")

    if len(sig) > 2:
        t = np.arange(len(sig))
        ax.plot(t, sig, color="#00e5cc", lw=1.5, alpha=0.9)
        ax.fill_between(t, sig.min(), sig, alpha=0.12, color="#00e5cc")

        # Mark every other reversal as a "rep" marker (every 2nd = full cycle top)
        if len(revs) >= 2:
            rep_revs = revs[::2]   # every other reversal = one full rep
            ax.scatter(rep_revs, sig[rep_revs], color="#ff2d6b", s=60,
                       zorder=5, marker="v",
                       label="{} rep{}".format(n_reps, "s" if n_reps!=1 else ""))
            ax.legend(facecolor="#111828", labelcolor="white",
                      fontsize=7, loc="upper left", framealpha=0.6)

        # Light vertical lines at ALL reversals (shows direction changes)
        for r in revs:
            ax.axvline(r, color="#ffffff", alpha=0.08, lw=0.8)

    blank = ("waiting", "analyzing", "")
    ex_str = exercise.replace("-"," ").replace("_"," ").upper() \
             if not any(b in exercise for b in blank) else "-"
    ax.set_title("{} | {} REPS | {}".format(ex_str, n_reps, key),
                 color="#c8d8f8", fontsize=7, pad=2, fontfamily="monospace")
    ax.set_xlim(0, max(len(sig)-1, 1))
    ax.tick_params(colors="#2a3a5a", labelsize=5)
    for sp in ax.spines.values(): sp.set_edgecolor("#111a30")

    fig.tight_layout(pad=0.3)
    img = _fig_to_bgr(fig)
    plt.close(fig)
    return img

# ─────────────────────────────────────────────────────────────────────────────
# Activity bars  (mini bar chart showing which joint groups are active)
# ─────────────────────────────────────────────────────────────────────────────
def draw_activity_bars(frame, activity_dict):
    """Draw 4 small activity bars in bottom-left corner."""
    h, w = frame.shape[:2]
    groups = ["arms", "legs", "core", "spread"]
    colors = [(0,200,255),(0,255,128),(255,180,0),(200,100,255)]
    bar_w, bar_h_max, gap, x0, y0 = 18, 60, 6, 12, h-75
    for i, (g, col) in enumerate(zip(groups, colors)):
        level  = activity_dict.get(g, 0.0)
        filled = int(level * bar_h_max)
        bx     = x0 + i*(bar_w + gap)
        # background
        cv2.rectangle(frame, (bx, y0), (bx+bar_w, y0+bar_h_max),
                      (20,20,30), -1)
        # filled
        if filled > 0:
            cv2.rectangle(frame, (bx, y0+bar_h_max-filled),
                          (bx+bar_w, y0+bar_h_max), col, -1)
        # label
        cv2.putText(frame, g[:3], (bx, y0+bar_h_max+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (80,90,120), 1, cv2.LINE_AA)


def _angle(a, b, c):
    ba = a - b
    bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _extract_classifier_features(landmarks):
    """
    Build one feature row matching train_classifier.py merged/renamed feature
    schema: landmarks__, angles__, dist3d__, distxyz__.
    """
    if landmarks is None:
        return None

    lm_list = landmarks.landmark if hasattr(landmarks, "landmark") else landmarks
    if lm_list is None or len(lm_list) < 33:
        return None

    p = np.array([[l.x, l.y, l.z] for l in lm_list], dtype=np.float32)

    row = {}
    for i, name in enumerate(MP_LANDMARKS):
        row["landmarks__x_{}".format(name)] = float(p[i, 0])
        row["landmarks__y_{}".format(name)] = float(p[i, 1])
        row["landmarks__z_{}".format(name)] = float(p[i, 2])

    angles = {
        "left_elbow_angle": _angle(p[11], p[13], p[15]),
        "right_elbow_angle": _angle(p[12], p[14], p[16]),
        "left_knee_angle": _angle(p[23], p[25], p[27]),
        "right_knee_angle": _angle(p[24], p[26], p[28]),
        "left_hip_angle": _angle(p[11], p[23], p[25]),
        "right_hip_angle": _angle(p[12], p[24], p[26]),
        "left_shoulder_angle": _angle(p[13], p[11], p[23]),
        "right_shoulder_angle": _angle(p[14], p[12], p[24]),
        "trunk_angle": _angle(p[25], p[23], p[11]),
        "neck_angle": _angle(p[11], p[0], p[12]),
    }
    for k, v in angles.items():
        row["angles__{}".format(k)] = float(v)

    def d3(i, j):
        return float(np.linalg.norm(p[i] - p[j]))

    dist3d = {
        "wrist_dist": d3(15, 16),
        "ankle_dist": d3(27, 28),
        "shoulder_dist": d3(11, 12),
        "hip_dist": d3(23, 24),
        "l_wrist_hip": d3(15, 23),
        "r_wrist_hip": d3(16, 24),
        "l_hand_head": d3(15, 0),
        "r_hand_head": d3(16, 0),
        "l_wrist_shoulder": d3(15, 11),
        "r_wrist_shoulder": d3(16, 12),
    }
    for k, v in dist3d.items():
        row["dist3d__{}".format(k)] = float(v)

    for name, i, j in (("wrist", 15, 16), ("ankle", 27, 28), ("hip", 23, 24)):
        diff = np.abs(p[i] - p[j])
        row["distxyz__{}_x".format(name)] = float(diff[0])
        row["distxyz__{}_y".format(name)] = float(diff[1])
        row["distxyz__{}_z".format(name)] = float(diff[2])

    return row


class StageRepCounter:
    """Rep counter using canonical stage transitions: down -> up."""

    def __init__(self):
        self.reps_by_exercise = collections.defaultdict(int)
        self._last_stage_by_exercise = {}
        self._seen_down_by_exercise = collections.defaultdict(bool)

    def reset(self):
        self.reps_by_exercise.clear()
        self._last_stage_by_exercise.clear()
        self._seen_down_by_exercise.clear()

    def update(self, exercise, stage):
        if not exercise or exercise == "unknown" or stage not in ("up", "down"):
            return
        prev = self._last_stage_by_exercise.get(exercise)
        if prev == stage:
            return
        self._last_stage_by_exercise[exercise] = stage
        if stage == "down":
            self._seen_down_by_exercise[exercise] = True
        elif stage == "up" and self._seen_down_by_exercise[exercise]:
            self.reps_by_exercise[exercise] += 1
            self._seen_down_by_exercise[exercise] = False

    def reps_for(self, exercise):
        return int(self.reps_by_exercise.get(exercise or "", 0))

# ─────────────────────────────────────────────────────────────────────────────
# HUD overlay
# ─────────────────────────────────────────────────────────────────────────────
CYAN=(230,210,10); GREEN=(60,220,60); YELLOW=(0,200,230); DIM=(80,90,120)

def draw_hud(frame, n_reps, result, llm_busy, fps, triggered):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (w,68), (4,6,14), -1)
    cv2.addWeighted(ov, 0.72, frame, 0.28, 0, frame)

    # Rep counter
    cv2.putText(frame, str(n_reps), (14,56),
                cv2.FONT_HERSHEY_DUPLEX, 2.1, (0,245,190), 3, cv2.LINE_AA)
    cv2.putText(frame, "REPS", (80,34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, DIM, 1, cv2.LINE_AA)

    # Exercise
    ex = result.exercise.upper().replace("-"," ").replace("_"," ") \
         if result.exercise else "-"
    cv2.putText(frame, ex, (148,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.82, CYAN, 1, cv2.LINE_AA)
    conf_col = {"high":GREEN,"medium":YELLOW,"low":DIM}.get(result.confidence,DIM)
    if result.confidence:
        cv2.putText(frame, result.confidence, (148,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, conf_col, 1, cv2.LINE_AA)

    # FPS + LLM status
    cv2.putText(frame, "{:.0f} fps".format(fps), (w-90,22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, DIM, 1, cv2.LINE_AA)
    if llm_busy:
        cv2.putText(frame, "LLM classifying...", (w-185,44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, YELLOW, 1, cv2.LINE_AA)
    elif triggered:
        cv2.putText(frame, "movement change!", (w-185,44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0,180,255), 1, cv2.LINE_AA)

    # Form tips
    for i, tip in enumerate(result.form_tips[:3]):
        cy  = 92 + i*22
        txt = tip if len(tip)<50 else tip[:47]+"..."
        cv2.putText(frame, "> "+txt, (w - min(w-10, len(txt)*9+24), cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, YELLOW, 1, cv2.LINE_AA)

    # Controls hint
    cv2.putText(frame, "[Q]uit [R]eset [C]lassify [S]ave",
                (10, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.32, DIM, 1, cv2.LINE_AA)

# ─────────────────────────────────────────────────────────────────────────────
# Save graph
# ─────────────────────────────────────────────────────────────────────────────
def save_graph(graph_data, exercise, n_reps, path="movement_graph.png"):
    fig, ax = plt.subplots(figsize=(13,4))
    fig.patch.set_facecolor("#07090f"); ax.set_facecolor("#07090f")
    sig, revs = graph_data["signal"], graph_data["reversals"]
    if len(sig) > 2:
        t = np.arange(len(sig))
        s = _smooth(sig)
        ax.plot(t, s, color="#00e5cc", lw=2)
        ax.fill_between(t, s.min(), s, alpha=0.18, color="#00e5cc")
        if len(revs) >= 2:
            rr = revs[::2]
            ax.scatter(rr, s[rr], color="#ff2d6b", s=100, zorder=5,
                       marker="v", label="{} reps".format(n_reps))
            ax.legend(facecolor="#111828", labelcolor="white")
        for r in revs:
            ax.axvline(r, color="#ffffff", alpha=0.08, lw=0.8)
    ex_str = exercise.upper().replace("-"," ").replace("_"," ")
    ax.set_title("Movement Graph -- {} | {} REPS".format(ex_str, n_reps),
                 color="#c8d8f8", fontsize=13)
    ax.set_xlabel("Frame", color="#5060a0")
    ax.set_ylabel("Joint angle (degrees)", color="#5060a0")
    ax.tick_params(colors="#2a3a5a")
    for sp in ax.spines.values(): sp.set_edgecolor("#111a30")
    plt.tight_layout()
    plt.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print("  [saved] {}".format(path))

# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_source(source):
    try:
        return int(source), True
    except (TypeError, ValueError):
        return str(source), False


def _confidence_bucket(conf):
    if conf >= 0.80:
        return "high"
    if conf >= 0.55:
        return "medium"
    return "low"


def run(source, clf_path, model, output_path, no_llm):
    try:
        import mediapipe as mp
        from mediapipe.tasks.python.vision import (
            PoseLandmarker, PoseLandmarkerOptions, RunningMode)
        from mediapipe.tasks.python.core.base_options import BaseOptions
    except ImportError:
        raise ImportError("pip install mediapipe>=0.10")

    ensure_model()

    latest_lm = [None]
    lm_lock   = threading.Lock()

    def on_result(det, image, ts):
        lms = det.pose_landmarks
        with lm_lock:
            latest_lm[0] = lms[0] if lms else None

    opts = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.LIVE_STREAM,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=on_result,
    )

    clf_model = None
    if clf_path:
        if not os.path.exists(clf_path):
            raise FileNotFoundError("Classifier model not found: {}".format(clf_path))
        clf_model = load_model(clf_path)
        print("Classifier: loaded {}".format(clf_path))
    else:
        print("Classifier: disabled (--clf not provided)")

    llm = None if no_llm else LLMClassifier(model=model)
    if llm:
        ok, msg = llm.check_connection()
        print("Ollama:", msg)
        if not ok:
            print("  -> Continuing without LLM")
            llm = None

    src_value, is_cam = _resolve_source(source)
    cap = cv2.VideoCapture(src_value)
    if not cap.isOpened():
        raise RuntimeError("Cannot open source {}".format(source))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output_path:
        fps_out = cap.get(cv2.CAP_PROP_FPS) or 30
        writer  = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                  fps_out, (fw, fh+160))

    analyzer     = MotionAnalyzer()
    stage_counter = StageRepCounter()
    result_obj   = ClassificationResult(exercise="waiting...")
    exercise     = "unknown"
    stage        = "unknown"
    frame_idx    = 0
    t0           = time.time()
    triggered    = False
    ex_hist      = collections.deque(maxlen=12)
    st_hist      = collections.deque(maxlen=8)
    conf_hist    = collections.deque(maxlen=20)

    cv2.namedWindow("RepCounter", cv2.WINDOW_NORMAL)
    print("\n>  Running  --  Q quit | R reset | C classify now | S save graph\n")

    with PoseLandmarker.create_from_options(opts) as lander:
        while True:
            ret, frame = cap.read()
            if not ret:
                if is_cam:
                    time.sleep(0.01)
                    continue
                break

            frame_idx += 1
            fps        = frame_idx / max(time.time() - t0, 0.01)

            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            lander.detect_async(mp_img, frame_idx)   # strictly monotonic ts

            with lm_lock:
                lm = latest_lm[0]

            draw_skeleton(frame, lm)
            analyzer.update(lm)

            # Primary classifier path: per-frame stage prediction, then smooth.
            if clf_model and lm is not None:
                row = _extract_classifier_features(lm)
                if row is not None:
                    try:
                        pred_label, pred_conf, _ = predict_pose(clf_model, row)
                        pred_label = normalize_label(pred_label)
                        pred_ex, pred_stage = split_label(pred_label)
                        if pred_ex != "unknown":
                            ex_hist.append(pred_ex)
                        if pred_stage in ("up", "down", "rest"):
                            st_hist.append(pred_stage)
                        conf_hist.append(float(pred_conf))
                    except Exception as e:
                        # Keep the stream alive if one frame fails inference.
                        print("  [classifier warning]", e)

            if ex_hist:
                exercise = max(set(ex_hist), key=ex_hist.count)
            if st_hist:
                stage = max(set(st_hist), key=st_hist.count)
            if exercise != "unknown":
                stage_counter.update(exercise, stage)

            signal_reps, sig_name, _ = analyzer.rep_count(exercise)
            stage_reps = stage_counter.reps_for(exercise)
            n_reps = stage_reps if stage in ("up", "down") else signal_reps

            if clf_model and exercise != "unknown":
                avg_conf = float(np.mean(conf_hist)) if conf_hist else 0.0
                disp_label = "{}_{}".format(exercise, stage) if stage in ("up", "down", "rest") else exercise
                result_obj.exercise = disp_label
                result_obj.confidence = _confidence_bucket(avg_conf)
                result_obj.error = ""

            # Optional LLM enriches tips/reasoning and can classify when no .pkl exists.
            if llm and llm.result.is_valid:
                llm_result = llm.result
                if not clf_model:
                    result_obj.exercise = llm_result.exercise
                    result_obj.confidence = llm_result.confidence
                    llm_ex, llm_stage = split_label(normalize_label(llm_result.exercise))
                    if llm_ex != "unknown":
                        exercise = llm_ex
                        if llm_stage != "unknown":
                            stage = llm_stage
                            stage_counter.update(exercise, stage)
                            n_reps = stage_counter.reps_for(exercise)
                result_obj.reasoning = llm_result.reasoning
                result_obj.form_tips = llm_result.form_tips

            graph_data           = analyzer.get_graph_data(exercise)
            activity             = analyzer.activity_summary()

            # Check for exercise change -> trigger LLM
            triggered = False
            if llm and not llm.busy:
                if analyzer.should_reclassify():
                    summary = analyzer.summarize()
                    print("\n[Activity change detected] Sending to LLM...\n")
                    llm.request(summary)
                    triggered = True

            draw_hud(frame, n_reps, result_obj, 
                     llm_busy=(llm is not None and llm.busy),
                     fps=fps, triggered=triggered)
            draw_activity_bars(frame, activity)

            graph_img = render_graph(graph_data, exercise, n_reps, fw, 160)
            if graph_img.shape[1] != fw:
                graph_img = cv2.resize(graph_img, (fw, 160))
            combined = np.vstack([frame, graph_img])

            cv2.imshow("RepCounter", combined)
            if writer:
                writer.write(combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                analyzer      = MotionAnalyzer()
                stage_counter = StageRepCounter()
                result_obj    = ClassificationResult(exercise="waiting...")
                exercise, stage = "unknown", "unknown"
                ex_hist.clear(); st_hist.clear(); conf_hist.clear()
                print("  [reset]")
            elif key == ord("c"):
                if llm and not llm.busy:
                    analyzer.force_reclassify()
                    summary = analyzer.summarize()
                    print("\n-- Summary --\n{}\n-------------\n".format(summary))
                    llm.request(summary)
            elif key == ord("s"):
                save_graph(graph_data, exercise, n_reps)

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()

    gd = analyzer.get_graph_data(exercise)
    n, _, _ = analyzer.rep_count(exercise)
    save_graph(gd, exercise, n, "movement_graph_final.png")
    print("\n-- Final summary --")
    print(analyzer.summarize())
    if result_obj.is_valid:
        print("\nFinal: {} ({}) — {}".format(
            result_obj.exercise, result_obj.confidence, result_obj.reasoning))
        for tip in result_obj.form_tips:
            print(" >", tip)

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source",  default="0", help="Webcam index or video path")
    p.add_argument("--clf",     default=None, help="Classifier .pkl path")
    p.add_argument("--model",   default="qwen3")
    p.add_argument("--output",  default=None)
    p.add_argument("--no_llm",  action="store_true")
    args = p.parse_args()
    run(source=args.source, clf_path=args.clf, model=args.model,
        output_path=args.output, no_llm=args.no_llm)
