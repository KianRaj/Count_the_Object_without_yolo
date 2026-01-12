# Vehicle Counting System - Vehant Hackathon

## Overview
This solution provides a robust, classical computer vision pipeline to count vehicles moving away from a static camera. It avoids deep learning, relying instead on temporal motion analysis, background modeling, and state-estimation tracking.

### Design Philosophy

Frame-by-frame detection leads to:
- Duplicate IDs
- Over-counting
- Sensitivity to shadows and noise

This system instead relies on **temporal consistency**:
a vehicle is defined by its motion footprint over time, not by a single frame.

### Counting Guarantee

A vehicle is counted **exactly once** if:
1. It exhibits consistent motion over multiple frames
2. It crosses a spatial boundary (line or zone)
3. It is not spatially or temporally overlapping with an already-counted track

Once counted, a track and its spatial neighbors are locked to prevent duplication.

### Why Blob Fusion Is Necessary

Large vehicles (e.g., trucks with loads) often appear as multiple motion blobs:
- Cabin
- Load
- Wheels

Counting these independently leads to over-counting.
The fusion stage merges nearby blobs into a single logical vehicle before tracking.

### Failure Modes & Mitigations

| Scenario | Mitigation |
|--------|------------|
Initial green jitter | Motion history + delayed counting |
Truck split into parts | Blob fusion + track deduplication |
Parallel vehicles | Independent Kalman tracks |
Shadows | Stationary filtering |
Slow vehicles | Soft motion thresholds |


## 🛠 The Pipeline
Our system follows a five-stage processing architecture:

### 1. Background Modeling & Motion Detection
We use **Gaussian Mixture-based Background/Foreground Segmentation (MOG2)**. 
- **Learning Rate:** Optimized to ignore gradual lighting changes.
- **Stability Guard:** A `movement_ratio` check ensures that if the frame is static (less than 0.05% change), the counting logic idles to prevent "ghost counts" from sensor noise.

### 2. Morphological Refinement
Raw masks are processed using **Opening and Closing (Elliptical Kernels)** to:
- Remove small noise (salt-and-pepper).
- Fuse fragmented blobs (e.g., a car body and its shadow).
- Close gaps in large vehicles (trucks/buses).

### 3. Feature-Based Tracking
- **Kalman Filter:** Every vehicle is assigned a 4-state Kalman Filter (position + velocity). This handles temporary occlusions and maintains trajectory even if the detector misses a frame.
- **Hungarian Algorithm:** Data association is performed via `linear_sum_assignment`, matching new detections to existing tracks by minimizing Euclidean distance.

### 4. Visual Reasoning & Filtering
- **Aspect Ratio Analysis:** Detections are filtered to reject tall, thin shapes (likely pedestrians) and extremely flat shapes (noise).
- **Displacement Validation:** A track is only considered "valid" for counting if it exhibits a minimum cumulative displacement. This prevents stationary objects near the counting line from triggering false positives.

### 5. Counting Logic
Vehicles are moving **away** from the camera (increasing Y-coordinate). The system utilizes:
- **Horizontal Line Cross:** Triggers when the centroid transitions over the 50% height mark.
- **Zone Entry:** Acts as a secondary check to catch vehicles moving at high velocity between frames.

## Setup & Execution
1. Install dependencies: `pip install -r requirements.txt`
2. Run the main script: `python main.py`

## Stability Features
- **Zero-Activity Reset:** If movement falls below a threshold, the system force-clears active tracks.
- **De-duplication:** Overlapping tracks are merged to prevent double-counting a single large vehicle.

## flowchart TD
    A[Input Video] --> B[Background Subtraction]
    B --> C[Morphological Filtering]
    C --> D[Contour Detection]
    D --> E[Blob Fusion]
    E --> F[Kalman Tracking]
    F --> G[Motion Analysis]
    G --> H{Valid Vehicle?}
    H -- Yes --> I[Line / Zone Crossing]
    I --> J[Count Vehicle]
    H -- No --> K[Ignore]

### Debug vs Evaluation Mode

- `debug=True`  
  Shows visual overlays, trajectories, diagnostics  
  Intended for development and tuning

Green   → Active Track
Blue    → Counted Vehicle
Purple  → Stationary / Rejected
Yellow  → Fused Detection
Orange  → Raw Detection


- `debug=False`  
  No visualization  
  Pure counting logic  
  Intended for final evaluation and submission


