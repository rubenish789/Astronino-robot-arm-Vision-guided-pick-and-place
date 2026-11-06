# Astorino Robot Arm — Vision-guided Pick and Place

A Kawasaki **astorino** 6-axis robot arm with a pneumatic gripper and an **Intel RealSense D435** depth camera mounted on the wrist (custom 3D-printed bracket). The robot finds colored cubes scattered on the table and stacks them into **four towers by color** — red, green, blue and yellow. When the build is done it keeps watching the table and **rebuilds the towers automatically** if someone knocks them over.

## Features

- **Eye-in-hand vision** — RealSense D435 on the wrist; color + depth frames from one fixed observation pose.
- **Table plane fitting** (RANSAC) and a **height map** above the table, so cubes, piles and towers are measured in 3D.
- **Color classification** in HSV with tunable thresholds (`hsv_tune.py`).
- **Touching / stacked cubes** — blobs are split along thin "necks" (corner contacts), the rest is divided with k-means (up to 6 cubes per pile).
- **Grasp planning** — the gripper is rotated parallel to the cube edges; out of 4 rotations the planner picks one where the fingers don't hit neighbours and the camera doesn't hit the towers.
- **Push planning** — when no cube can be picked safely, one finger pushes a cube out of a pile. 8 directions × several lengths are simulated, predicting where neighbouring cubes will end up.
- **Watch mode** — after the build, the camera monitors the table; it waits while a human hand is in the frame and starts rebuilding a few seconds after the scene is stable.
- **Camera → robot calibration** — hand-taught points + Kabsch (SVD) rigid fit, RMS ≈ 3.5 mm.
- **Native USB driver** for the astorino controller (binary protocol with CRC, keep-alive thread, auto-reconnect).

## How one cycle works

```
look()  : snapshot → vision (plane, height map, cube faces) → scene.analyze()
plan()  : planner.clean_action / next_action → Action (pick / place / push / clean)
execute : pick_place.execute → astorino_usb (robot motion)
```

1. Take a new snapshot, find cubes, measure tower heights, split piles.
2. If cubes block a tower spot — clear it first.
3. Otherwise pick the first cube that can be grasped safely and place it on the tower of its color.
4. If nothing can be picked — choose the best push.

## Project structure

```
├── calibration.json            camera → robot calibration (R, t, observation pose)
├── vision_detect.py            standalone RealSense + HSV cube detection test
└── astorino_vision/
    ├── config.py               all settings: tower spots, grasp heights, speeds, HSV ranges
    ├── pick_place.py           main program: robot motions, build loop, UI window, watch mode
    ├── planner.py              decides what to do next: grasp pose, push, clearing tower spots
    ├── scene.py                parses a snapshot: loose cubes, piles, tower heights
    ├── vision.py               RealSense camera, table plane, height map, cube detection
    ├── colors.py               HSV color classification
    ├── geometry.py             calibration, rigid fit, rotations, collision rectangles, push prediction
    ├── astorino_usb.py         USB driver for the astorino controller
    ├── calibrate.py            interactive camera → robot calibration
    ├── hsv_tune.py             HSV threshold tuning with sliders
    └── NOTES.md                detailed notes (in Russian): tuning history, known issues
```

## Hardware

- Kawasaki astorino robot arm (controller connected over USB)
- Pneumatic two-finger gripper
- Intel RealSense D435 on a custom 3D-printed wrist mount
- Colored 3D cubes

## Run

```bash
pip install -r astorino_vision/requirements.txt
# from the repository root:
python -m astorino_vision.calibrate          # calibrate camera → robot (once / after moving the camera)
python -m astorino_vision.pick_place         # start; press A in the camera window to build all towers
python -m astorino_vision.pick_place --watch # watch mode on from the start
python -m astorino_vision.pick_place --step  # confirm every motion with Enter
```

Keys in the camera window: `A` build all · `SPACE` one cycle · `W` toggle watch · `R` re-measure towers · `S` stop after current action · `Q`/`Esc` quit.

The robot must be in REPEAT mode, zeroed, with no errors on the pendant; close astorino software / RoboDK so the USB port is free.

## Tech stack

Python · OpenCV · NumPy · pyrealsense2 · pyserial
