# 🏋️ Exercise Rep Counter — MediaPipe + Classifier

Real-time **rep counting · exercise classification · form feedback · movement graph**  
for 4 exercises: Squats · Push-ups · Pull-ups · Jumping Jacks

# LLM BRANCH

To run the code with an LLM use the ```llmrepcounter.py```
The ```rep_counter.py``` also has it, however it has the other trained classifier as the experiment too

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
0,jumpingjack_down
1,jumpingjack_up
2,squat_down
...
```
Canonical classes follow `<exercise>_<stage>`:
`jumpingjack_up/down/rest`, `pushup_up/down/rest`, `pullup_up/down/rest`,
`squat_up/down/rest`

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
| squat         | left knee Y   | down→up transitions |
| pushup        | left elbow Y  | down→up transitions |
| pullup        | left wrist Y  | down→up transitions |
| jumpingjack   | left wrist Y  | down→up transitions |

Per-frame stage prediction is smoothed and reps are counted via stage transitions (`up → down → up`).
Signal smoothing + `scipy.find_peaks` is still rendered for the movement graph.

### Form Feedback
Real-time joint angle checks displayed as colour-coded overlays:
- 🟢 Green = correct
- 🔴 Red = needs correction

| Exercise      | Checks |
|--------------|--------|
| squat         | depth, torso lean, knee tracking |
| pushup        | arm extension, back sag, range |
| pullup        | chin height, bottom extension |
| jumpingjack   | arm height, leg spread |

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
