"""
rep_counter.py
==============
Live webcam rep counter + Ollama exercise classifier.

  MediaPipe Pose -> MotionAnalyzer (zero-crossing rep count)
                -> Activity fingerprint change detection
                -> LLMClassifier (Ollama) on exercise change
                -> HUD + movement graph

Usage:
  python rep_counter.py
  python rep_counter.py --model qwen3
  python rep_counter.py --no_llm
  python rep_counter.py --output out.mp4

Controls:  Q quit | R reset | C classify now | S save graph
"""

import argparse, os, time, threading, urllib.request
import cv2, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from motion_analyzer import MotionAnalyzer, _smooth, EXERCISE_SIGNAL
from llm_classifier  import LLMClassifier, ClassificationResult

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
def run(model, output_path, no_llm, cam_index):
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

    llm = None if no_llm else LLMClassifier(model=model)
    if llm:
        ok, msg = llm.check_connection()
        print("Ollama:", msg)
        if not ok:
            print("  -> Continuing without LLM")
            llm = None

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera {}".format(cam_index))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output_path:
        fps_out = cap.get(cv2.CAP_PROP_FPS) or 30
        writer  = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                  fps_out, (fw, fh+160))

    analyzer   = MotionAnalyzer()
    result_obj = ClassificationResult()
    exercise   = "waiting..."
    frame_idx  = 0
    t0         = time.time()
    triggered  = False   # flag to show "movement change!" in HUD

    cv2.namedWindow("RepCounter", cv2.WINDOW_NORMAL)
    print("\n>  Running  --  Q quit | R reset | C classify now | S save graph\n")

    with PoseLandmarker.create_from_options(opts) as lander:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame_idx += 1
            fps        = frame_idx / max(time.time() - t0, 0.01)

            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            lander.detect_async(mp_img, frame_idx)   # strictly monotonic ts

            with lm_lock:
                lm = latest_lm[0]

            draw_skeleton(frame, lm)
            analyzer.update(lm)

            # Update exercise from LLM result + use its recommended signal
            if llm and llm.result.is_valid:
                result_obj = llm.result
                exercise   = result_obj.exercise
                # Override EXERCISE_SIGNAL with what the LLM recommended
                if result_obj.rep_signal:
                    from motion_analyzer import EXERCISE_SIGNAL
                    EXERCISE_SIGNAL[exercise.lower()] = result_obj.rep_signal

            n_reps, sig_name, _ = analyzer.rep_count(exercise)
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
                analyzer   = MotionAnalyzer()
                result_obj = ClassificationResult()
                exercise   = "waiting..."
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
    p.add_argument("--model",   default="qwen3")
    p.add_argument("--output",  default=None)
    p.add_argument("--no_llm",  action="store_true")
    p.add_argument("--cam",     type=int, default=0)
    args = p.parse_args()
    run(model=args.model, output_path=args.output,
        no_llm=args.no_llm, cam_index=args.cam)
