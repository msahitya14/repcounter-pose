"""
rep_counter.py
==============
Live webcam rep counter pipeline:

  MediaPipe Pose  ->  MotionAnalyzer  ->  rep count + movement graph
                  ->  (every N secs)  ->  LLMClassifier (Ollama)
                                      ->  exercise label + form tips

Uses the MediaPipe Tasks API (mediapipe >= 0.10) which replaced solutions.pose.
A ~5 MB model file is downloaded automatically on first run.

Usage:
  python rep_counter.py
  python rep_counter.py --model mistral
  python rep_counter.py --model qwen3 --classify_every 8
  python rep_counter.py --no_llm        # pure rep counter, no classification
  python rep_counter.py --output out.mp4

Controls (while running):
  Q   quit
  R   reset rep buffer / motion history
  C   force LLM classification right now
  S   save current movement graph as PNG
"""

import argparse
import os
import time
import threading
import urllib.request
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from motion_analyzer import MotionAnalyzer, _smooth, _count_peaks
from llm_classifier  import LLMClassifier, ClassificationResult

# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe model download (Tasks API needs a .task file)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)
MODEL_PATH = "pose_landmarker.task"


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading MediaPipe pose model (~5 MB) ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("  saved -> {}".format(MODEL_PATH))


# ─────────────────────────────────────────────────────────────────────────────
# Skeleton drawing  (Tasks API landmarks = plain list, no .visibility attr)
# ─────────────────────────────────────────────────────────────────────────────
MP_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (0,  11), (0,  12),
]


def draw_skeleton(frame, landmarks, conf_thr=0.3):
    """landmarks: plain list[NormalizedLandmark] from Tasks API."""
    if not landmarks:
        return
    h, w = frame.shape[:2]

    def get_vis(lm):
        return getattr(lm, "presence", getattr(lm, "visibility", 1.0))

    pts = np.array([[l.x * w, l.y * h] for l in landmarks])
    v   = np.array([get_vis(l) for l in landmarks])

    for a, b in MP_CONNECTIONS:
        if a < len(pts) and b < len(pts) and v[a] > conf_thr and v[b] > conf_thr:
            cv2.line(frame, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)),
                     (255, 200, 0), 2, cv2.LINE_AA)
    for i in range(len(pts)):
        if v[i] > conf_thr:
            cv2.circle(frame, tuple(pts[i].astype(int)),
                       4, (0, 255, 128), -1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Movement graph renderer
# ─────────────────────────────────────────────────────────────────────────────
def render_graph(graph_data, exercise, n_reps, width, height=150):
    dpi = 80
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.patch.set_facecolor("#07090f")
    ax.set_facecolor("#07090f")

    sig   = graph_data["signal"]
    peaks = graph_data["peaks"]
    key   = graph_data["key"].replace("_", " ")

    if len(sig) > 2:
        t = np.arange(len(sig))
        ax.plot(t, sig, color="#00e5cc", lw=1.6, alpha=0.95)
        ax.fill_between(t, sig.min(), sig, alpha=0.14, color="#00e5cc")
        if len(peaks):
            label = "{} rep{}".format(n_reps, "s" if n_reps != 1 else "")
            ax.scatter(peaks, sig[peaks], color="#ff2d6b", s=55, zorder=5,
                       marker="v", label=label)
            ax.legend(facecolor="#111828", labelcolor="white",
                      fontsize=7, loc="upper left", framealpha=0.6, edgecolor="#223")

    blank = ("analyzing", "waiting", "")
    ex_str = (exercise.replace("_", " ").upper()
              if not any(b in exercise for b in blank) else "-")
    rep_str = "REP" if n_reps == 1 else "REPS"
    ax.set_title(
        "{}   *   {} {}   *   signal: {}".format(ex_str, n_reps, rep_str, key),
        color="#c8d8f8", fontsize=7, pad=2, fontfamily="monospace",
    )
    ax.set_xlim(0, max(len(sig) - 1, 1))
    ax.tick_params(colors="#2a3a5a", labelsize=5)
    for sp in ax.spines.values():
        sp.set_edgecolor("#111a30")

    fig.tight_layout(pad=0.3)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# ─────────────────────────────────────────────────────────────────────────────
# HUD overlay
# ─────────────────────────────────────────────────────────────────────────────
CYAN   = (230, 210,  10)
GREEN  = ( 60, 220,  60)
YELLOW = (  0, 200, 230)
DIM    = ( 80,  90, 120)


def draw_hud(frame, n_reps, result, llm_busy, fps, classify_in):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 68), (4, 6, 14), -1)
    cv2.addWeighted(ov, 0.72, frame, 0.28, 0, frame)

    cv2.putText(frame, str(n_reps), (14, 56),
                cv2.FONT_HERSHEY_DUPLEX, 2.1, (0, 245, 190), 3, cv2.LINE_AA)
    cv2.putText(frame, "REPS", (80, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, DIM, 1, cv2.LINE_AA)

    ex_text = result.exercise.upper().replace("_", " ") if result.exercise else "-"
    cv2.putText(frame, ex_text, (148, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.82, CYAN, 1, cv2.LINE_AA)

    conf_color = {"high": GREEN, "medium": YELLOW, "low": DIM}.get(result.confidence, DIM)
    if result.confidence:
        cv2.putText(frame, result.confidence, (148, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, conf_color, 1, cv2.LINE_AA)

    cv2.putText(frame, "{:.0f} fps".format(fps), (w - 90, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, DIM, 1, cv2.LINE_AA)

    if llm_busy:
        cv2.putText(frame, "LLM thinking...", (w - 165, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, YELLOW, 1, cv2.LINE_AA)
    elif classify_in > 0:
        cv2.putText(frame, "classify in {:.0f}s".format(classify_in),
                    (w - 180, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.38, DIM, 1, cv2.LINE_AA)

    for i, tip in enumerate(result.form_tips[:3]):
        cy = 95 + i * 24
        short = tip if len(tip) < 52 else tip[:49] + "..."
        cx = max(10, w - len(short) * 9 - 20)
        cv2.putText(frame, "> " + short, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, YELLOW, 1, cv2.LINE_AA)

    cv2.putText(frame, "[Q]uit  [R]eset  [C]lassify now  [S]ave graph",
                (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, DIM, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Save graph PNG
# ─────────────────────────────────────────────────────────────────────────────
def save_graph(graph_data, exercise, n_reps, path="movement_graph.png"):
    fig, ax = plt.subplots(figsize=(13, 4))
    fig.patch.set_facecolor("#07090f")
    ax.set_facecolor("#07090f")
    sig   = graph_data["signal"]
    peaks = graph_data["peaks"]
    if len(sig) > 2:
        t = np.arange(len(sig))
        s = _smooth(sig)
        ax.plot(t, s, color="#00e5cc", lw=2.2)
        ax.fill_between(t, s.min(), s, alpha=0.18, color="#00e5cc")
        if len(peaks):
            ax.scatter(peaks, s[peaks], color="#ff2d6b", s=100, zorder=5,
                       marker="v", label="{} reps".format(n_reps))
            ax.legend(facecolor="#111828", labelcolor="white")
    ex_str = exercise.upper().replace("_", " ")
    ax.set_title("Movement Graph -- {}  *  {} REPS".format(ex_str, n_reps),
                 color="#c8d8f8", fontsize=13)
    ax.set_xlabel("Frame", color="#5060a0")
    ax.set_ylabel("Joint position (normalised)", color="#5060a0")
    ax.tick_params(colors="#2a3a5a")
    for sp in ax.spines.values():
        sp.set_edgecolor("#111a30")
    plt.tight_layout()
    plt.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print("  [saved] {}".format(path))


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
def run(model, classify_every, output_path, no_llm, cam_index):
    try:
        import mediapipe as mp
        from mediapipe.tasks.python.vision import (
            PoseLandmarker, PoseLandmarkerOptions, RunningMode
        )
        from mediapipe.tasks.python.core.base_options import BaseOptions
    except ImportError:
        raise ImportError("pip install mediapipe>=0.10")

    ensure_model()

    # Shared latest landmarks — written by callback, read by main loop
    latest_landmarks = [None]
    result_lock = threading.Lock()

    def on_result(detection_result, image, timestamp_ms):
        lms = detection_result.pose_landmarks
        with result_lock:
            latest_landmarks[0] = lms[0] if lms else None

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.LIVE_STREAM,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=on_result,
    )

    # Ollama setup
    llm = None if no_llm else LLMClassifier(model=model)
    if llm:
        ok, msg = llm.check_connection()
        print("Ollama:", msg)
        if not ok:
            print("  -> Starting without LLM (use --no_llm to suppress)")
            llm = None

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera {}".format(cam_index))

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output_path:
        fps_out = cap.get(cv2.CAP_PROP_FPS) or 30
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps_out, (fw, fh + 150))

    analyzer        = MotionAnalyzer()
    result_obj      = ClassificationResult()
    last_classify_t = time.time()
    frame_idx       = 0
    t0              = time.time()

    print("\n>  Running -- Q quit | R reset | C classify now | S save graph\n")

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            fps          = frame_idx / max(time.time() - t0, 0.01)
            timestamp_ms = int((time.time() - t0) * 1000)

            # Send frame to MediaPipe (result arrives via callback)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
            landmarker.detect_async(mp_image, timestamp_ms)

            # Read latest landmarks
            with result_lock:
                lm = latest_landmarks[0]

            draw_skeleton(frame, lm)
            analyzer.update(lm)

            n_reps, _  = analyzer.rep_count()
            graph_data = analyzer.get_graph_data()

            # Periodic LLM classification
            now         = time.time()
            classify_in = max(0.0, classify_every - (now - last_classify_t))

            if llm:
                if classify_in == 0 and not llm.busy and analyzer.frame_count > 60:
                    llm.request(analyzer.summarize())
                    last_classify_t = now
                if llm.result.is_valid:
                    result_obj = llm.result

            exercise = result_obj.exercise

            draw_hud(frame, n_reps, result_obj,
                     llm_busy=(llm is not None and llm.busy),
                     fps=fps, classify_in=classify_in)

            graph_img = render_graph(graph_data, exercise, n_reps, fw, 150)
            if graph_img.shape[1] != fw:
                graph_img = cv2.resize(graph_img, (fw, 150))
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
                print("  [reset]")
            elif key == ord("c"):
                if llm and not llm.busy and analyzer.frame_count > 30:
                    summary = analyzer.summarize()
                    print("\n-- Movement summary --\n{}\n---------------------\n".format(summary))
                    llm.request(summary)
                    last_classify_t = now
            elif key == ord("s"):
                save_graph(graph_data, exercise, n_reps)

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    graph_data = analyzer.get_graph_data()
    n_reps, _  = analyzer.rep_count()
    save_graph(graph_data, result_obj.exercise, n_reps, "movement_graph_final.png")
    print("\n-- Final movement summary --")
    print(analyzer.summarize())
    if result_obj.is_valid:
        print("\nFinal: {} ({})".format(result_obj.exercise, result_obj.confidence))
        print("Reasoning:", result_obj.reasoning)
        for tip in result_obj.form_tips:
            print(" >", tip)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",          default="qwen3")
    p.add_argument("--classify_every", type=float, default=10.0)
    p.add_argument("--output",         default=None)
    p.add_argument("--no_llm",         action="store_true")
    p.add_argument("--cam",            type=int, default=0)
    args = p.parse_args()
    run(model=args.model, classify_every=args.classify_every,
        output_path=args.output, no_llm=args.no_llm, cam_index=args.cam)
