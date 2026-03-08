"""
prepare_data.py
================
Generates synthetic demo data in your EXACT CSV format so you can
test the pipeline before plugging in real recordings.

Your real data: just drop your actual CSVs into data/ with the same schema.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── MediaPipe landmark names (33) ──────────────────────────────────────────────
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

ANGLE_NAMES = [
    "left_elbow_angle","right_elbow_angle",
    "left_knee_angle","right_knee_angle",
    "left_hip_angle","right_hip_angle",
    "left_shoulder_angle","right_shoulder_angle",
    "trunk_angle","neck_angle",
]

DIST3D_NAMES = [
    "wrist_dist","ankle_dist","shoulder_dist","hip_dist",
    "l_wrist_hip","r_wrist_hip","l_hand_head","r_hand_head",
    "l_wrist_shoulder","r_wrist_shoulder",
]

DISTXYZ_NAMES = [f"{d}_{ax}" for d in ["wrist","ankle","hip"] for ax in ["x","y","z"]]


# ── Pose templates (normalised, roughly -100..100 scale like your data) ────────
def base_standing():
    p = np.zeros((33, 3))
    # Head cluster
    p[0]  = [0, -58, -45]     # nose
    p[1:5] = [[-4,-63,-44],[-3,-63,-44],[-2,-64,-44],[-7,-61,-43]]
    p[5:7] = [[-8,-61,-43],[-9,-61,-43]]
    p[7]  = [0,-63,-36]; p[8] = [-9,-59,-29]
    p[9]  = [-3,-54,-41]; p[10] = [-7,-53,-39]
    # Shoulders
    p[11] = [8,-41,-24]; p[12] = [-9,-39,-23]
    # Elbows
    p[13] = [10,-19,-11]; p[14] = [-12,-16,-14]
    # Wrists
    p[15] = [10,-5,-13]; p[16] = [-11,1,-20]
    # Hands
    p[17:23] = [[10,3,-15],[-12,7,-22],[10,2,-14],[-11,7,-22],[9,6,-25],[-10,6,-26]]
    # Hips
    p[23] = [5,0,-4]; p[24] = [-5,0,5]
    # Knees
    p[25] = [5,34,52]; p[26] = [-1,34,47]
    # Ankles
    p[27] = [4,28,57]; p[28] = [-1,35,51]
    # Heels / feet
    p[29] = [6,46,50]; p[30] = [0,46,41]
    p[31] = [6,46,50]; p[32] = [0,46,42]
    return p.astype(np.float32)


def gen_pose(exercise: str, phase: float) -> np.ndarray:
    """phase in [0,1]: 0=start/up, 1=bottom/down"""
    p = base_standing()
    t = phase

    if exercise == "squats":
        # Hips sink, knees bend
        p[23:29, 1] += t * 25
        p[25, 1] += t * 10; p[26, 1] += t * 10
        p[13, 1] += t * 5;  p[14, 1] += t * 5

    elif exercise == "pushups":
        # Whole body horizontal, elbow bend
        p[:, 1] += 30
        p[13, 1] += t * 12; p[14, 1] += t * 12
        p[15, 1] += t * 12; p[16, 1] += t * 12

    elif exercise == "pull_ups":
        # Wrists above head
        p[15, 1] -= (1 - t) * 40; p[16, 1] -= (1 - t) * 40
        p[13, 1] -= (1 - t) * 20; p[14, 1] -= (1 - t) * 20

    elif exercise == "situps":
        # Nose rises up
        p[0:12, 1] -= t * 30
        p[11, 1] -= t * 15; p[12, 1] -= t * 15

    elif exercise == "jumping_jacks":
        # Arms rise, legs spread
        spread = t
        p[15, 0] += spread * 40; p[16, 0] -= spread * 40
        p[15, 1] -= spread * 35; p[16, 1] -= spread * 35
        p[13, 0] += spread * 20; p[14, 0] -= spread * 20
        p[27, 0] += spread * 20; p[28, 0] -= spread * 20

    p += np.random.randn(*p.shape).astype(np.float32) * 0.8
    return p


def lm_to_angles(p):
    def ang(a, b, c):
        ba = p[a] - p[b]; bc = p[c] - p[b]
        cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    return {
        "left_elbow_angle":    ang(11, 13, 15),
        "right_elbow_angle":   ang(12, 14, 16),
        "left_knee_angle":     ang(23, 25, 27),
        "right_knee_angle":    ang(24, 26, 28),
        "left_hip_angle":      ang(11, 23, 25),
        "right_hip_angle":     ang(12, 24, 26),
        "left_shoulder_angle": ang(13, 11, 23),
        "right_shoulder_angle":ang(14, 12, 24),
        "trunk_angle":         ang(25, 23, 11),
        "neck_angle":          ang(11, 0, 12),
    }

def lm_to_dist3d(p):
    def d3(i, j): return float(np.linalg.norm(p[i] - p[j]))
    return {
        "wrist_dist": d3(15,16), "ankle_dist": d3(27,28),
        "shoulder_dist": d3(11,12), "hip_dist": d3(23,24),
        "l_wrist_hip": d3(15,23), "r_wrist_hip": d3(16,24),
        "l_hand_head": d3(15,0),  "r_hand_head": d3(16,0),
        "l_wrist_shoulder": d3(15,11), "r_wrist_shoulder": d3(16,12),
    }

def lm_to_distxyz(p):
    feats = {}
    for (name, i, j) in [("wrist",15,16),("ankle",27,28),("hip",23,24)]:
        diff = np.abs(p[i] - p[j])
        feats[f"{name}_x"] = float(diff[0])
        feats[f"{name}_y"] = float(diff[1])
        feats[f"{name}_z"] = float(diff[2])
    return feats


# ── Generate dataset ───────────────────────────────────────────────────────────
EXERCISES = ["squats","pushups","pull_ups","situps","jumping_jacks"]
STATES     = {"up": 0.0, "down": 1.0}

def create_dataset(out_dir="data", n_per_class=150):
    out = Path(out_dir); out.mkdir(exist_ok=True)

    lm_rows, ang_rows, dist3_rows, distxyz_rows, label_rows = [], [], [], [], []
    pid = 0

    for ex in EXERCISES:
        for state, phase_val in STATES.items():
            label = f"{ex}_{state}" if ex != "pushups" else f"pushups_{state}"
            for _ in range(n_per_class):
                phase = phase_val + np.random.uniform(-0.15, 0.15)
                phase = np.clip(phase, 0, 1)
                p = gen_pose(ex, phase)

                # landmarks row
                lm_row = {"pose_id": pid}
                for i, name in enumerate(MP_NAMES):
                    lm_row[f"x_{name}"] = p[i, 0]
                    lm_row[f"y_{name}"] = p[i, 1]
                    lm_row[f"z_{name}"] = p[i, 2]
                lm_rows.append(lm_row)

                ang_row = {"pose_id": pid}
                ang_row.update(lm_to_angles(p))
                ang_rows.append(ang_row)

                d3_row = {"pose_id": pid}
                d3_row.update(lm_to_dist3d(p))
                dist3_rows.append(d3_row)

                dxyz_row = {"pose_id": pid}
                dxyz_row.update(lm_to_distxyz(p))
                distxyz_rows.append(dxyz_row)

                label_rows.append({"pose_id": pid, "pose": label})
                pid += 1

    pd.DataFrame(lm_rows).to_csv(out / "landmarks.csv", index=False)
    pd.DataFrame(ang_rows).to_csv(out / "angles.csv", index=False)
    pd.DataFrame(dist3_rows).to_csv(out / "3d_distances.csv", index=False)
    pd.DataFrame(distxyz_rows).to_csv(out / "xyz_distances.csv", index=False)
    pd.DataFrame(label_rows).to_csv(out / "labels.csv", index=False)

    total = len(label_rows)
    print(f"✅ Demo dataset → {out_dir}/   ({total} poses, {len(EXERCISES)*2} classes)")
    df = pd.DataFrame(label_rows)
    print(df.groupby("pose")["pose_id"].count().to_string())


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data")
    p.add_argument("--n",   type=int, default=150)
    args = p.parse_args()
    create_dataset(args.out, args.n)
    print("\nNext:")
    print("  python train_classifier.py --data_dir data/ --model_out models/classifier.pkl")
    print("  python rep_counter.py --source 0 --clf models/classifier.pkl")
