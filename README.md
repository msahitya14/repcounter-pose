# 🏋️ Exercise Rep Counter — MediaPipe + Classifier

Real-time **rep counting · exercise classification · form feedback · movement graph**  
for 5 exercises: Squats · Push-ups · Pull-ups · Sit-ups · Jumping Jacks

---

## Quick Start

```bash
pip install -r requirements.txt

# 1. (optional) Generate synthetic demo data to test immediately
python prepare_data.py --out data/ --n 150

# 2. Train the classifier on your CSVs
python train_classifier.py --data_dir data/ --model_out models/classifier.pkl

# 3. Run live
python rep_counter.py --source 0 --clf models/classifier.pkl

# or on a video file
python rep_counter.py --source workout.mp4 --clf models/classifier.pkl --output out.mp4
```

---

## Your Data Format

All five CSVs share the same `pose_id` index (one row = one frame/pose).

### `labels.csv`
```
pose_id,pose
0,jumping_jacks_down
1,jumping_jacks_up
2,squats_down
...
```
**10 classes:** `jumping_jacks_up/down`, `pushups_up/down`, `pull_ups_up/down`,  
`situps_up/down`, `squats_up/down`

### `landmarks.csv`
33 MediaPipe landmarks × x, y, z  
Columns: `pose_id, x_nose, y_nose, z_nose, x_left_eye_inner, ...`

### `angles.csv`
Joint angles (degrees):  
`pose_id, left_elbow_angle, right_elbow_angle, left_knee_angle, right_knee_angle, ...`

### `3d_distances.csv`
3D Euclidean distances between landmark pairs:  
`pose_id, wrist_dist, ankle_dist, shoulder_dist, ...`

### `xyz_distances.csv`
Per-axis distances:  
`pose_id, wrist_x, wrist_y, wrist_z, ankle_x, ...`

---

## How It Works

### Classification
- All four feature CSVs are merged on `pose_id`
- **Gradient Boosting classifier** on the combined feature matrix  
  (~200+ features: raw landmarks + angles + distances)
- Per-frame prediction → smoothed exercise label → drives rep counting

### Rep Counting
| Exercise      | Signal joint  | Counts    |
|--------------|--------------|-----------|
| squats        | left knee Y   | valleys   |
| pushups       | left elbow Y  | peaks     |
| pull_ups      | left wrist Y  | peaks     |
| situps        | nose Y        | peaks     |
| jumping_jacks | left wrist Y  | peaks     |

Savitzky-Golay smoothing → `scipy.find_peaks` → rep count

### Form Feedback
Real-time joint angle checks displayed as colour-coded overlays:
- 🟢 Green = correct
- 🔴 Red = needs correction

| Exercise      | Checks |
|--------------|--------|
| squats        | depth, torso lean, knee tracking |
| pushups       | arm extension, back sag, range |
| pull_ups      | chin height, bottom extension |
| situps        | crunch depth, neck position |
| jumping_jacks | arm height, leg spread |

---

## Files

```
rep_counter_system/
├── train_classifier.py   ← Train on your 5 CSVs
├── rep_counter.py        ← Live webcam/video rep counter
├── prepare_data.py       ← Generate synthetic demo data
├── requirements.txt
└── README.md
```
