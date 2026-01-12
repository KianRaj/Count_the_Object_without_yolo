import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from IPython.display import clear_output
import time

# =========================
# KALMAN FILTER
# =========================
class Kalman:
    def __init__(self, x, y):
        self.x = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.P = np.eye(4) * 400
        self.F = np.array([[1,0,1,0],
                           [0,1,0,1],
                           [0,0,1,0],
                           [0,0,0,1]], dtype=np.float32)
        self.H = np.array([[1,0,0,0],
                           [0,1,0,0]], dtype=np.float32)
        self.R = np.eye(2) * 15
        self.Q = np.eye(4)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[0,0], self.x[1,0]

    def update(self, z):
        z = np.array(z, dtype=np.float32).reshape(2,1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P


# =========================
# TRACK OBJECT - ENHANCED
# =========================
class Track:
    def __init__(self, centroid, box, tid):
        self.kf = Kalman(*centroid)
        self.centroid = centroid
        self.prev_y = centroid[1]
        self.prev_x = centroid[0]
        self.box = box
        self.id = tid
        self.missed = 0
        self.counted = False
        self.age = 0
        self.positions = [centroid]
        self.direction = None
        self.crossed_horizontal = False
        self.crossed_vertical = False
        
        # Motion tracking
        self.first_position = centroid
        self.max_displacement = 0
        self.stationary_count = 0
        self.motion_score = 0  # Cumulative motion


# =========================
# IMPROVED BLOB FUSION
# =========================
def fuse_detections(dets, x_thresh=60, y_thresh=80):
    if not dets:
        return []
    
    fused = []
    used = [False]*len(dets)

    for i,(cx,cy,box) in enumerate(dets):
        if used[i]:
            continue

        x,y,w,h = box
        fx,fy,fw,fh = x,y,w,h
        merged_count = 1

        changed = True
        while changed:
            changed = False
            for j,(cx2,cy2,box2) in enumerate(dets):
                if i == j or used[j]:
                    continue
                
                x2,y2,w2,h2 = box2
                
                x_overlap = (fx < x2+w2) and (fx+fw > x2)
                y_overlap = (fy < y2+h2) and (fy+fh > y2)
                
                cx_curr = fx + fw/2
                cy_curr = fy + fh/2
                dist = np.hypot(cx_curr - cx2, cy_curr - cy2)
                
                should_merge = False
                
                if x_overlap and y_overlap:
                    should_merge = True
                elif dist < 100:
                    should_merge = True
                elif abs(cx_curr - cx2) < x_thresh and abs(cy_curr - cy2) < y_thresh:
                    should_merge = True
                
                if should_merge:
                    fx = min(fx, x2)
                    fy = min(fy, y2)
                    fw = max(fx+fw, x2+w2) - fx
                    fh = max(fy+fh, y2+h2) - fy
                    used[j] = True
                    merged_count += 1
                    changed = True

        used[i] = True
        if merged_count >= 2 or (fw * fh) > 1500:
            fused.append((fx+fw/2, fy+fh/2, (fx,fy,fw,fh)))

    return fused


# =========================
# SOFT FILTERS 
# =========================
def calculate_motion_metrics(track):
    """Calculate motion metrics without strict filtering"""
    if len(track.positions) < 2:
        return 0, 0, False
    
    # Total displacement from start
    start = np.array(track.first_position)
    current = np.array(track.centroid)
    total_displacement = np.linalg.norm(current - start)
    
    # Recent frame-to-frame movement
    recent_movement = 0
    if len(track.positions) >= 2:
        recent_movement = np.linalg.norm(
            np.array(track.positions[-1]) - np.array(track.positions[-2])
        )
    
    # Is likely moving (soft check)
    is_moving = total_displacement > 5 or recent_movement > 1.5
    
    return total_displacement, recent_movement, is_moving


def check_person_like(box, frame_shape):
    """
    Soft filter: Check if detection looks like a standing person
    Only reject OBVIOUS person shapes
    """
    h, w = frame_shape[:2]
    x, y, bw, bh = box
    
    aspect = bw / bh if bh > 0 else 1
    area = bw * bh
    
    # Only reject very obvious person shapes
    # Tall and thin (standing person)
    if aspect < 0.45 and bh > bw * 2.2 and area < 3000:
        return True
    
    # Very small objects (likely noise or small person)
    if area < 400:
        return True
    
    return False


# =========================
# SOLUTION CLASS - BALANCED
# =========================
class Solution:
    def forward(self, video_path, debug=True):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if not ret:
            return 0

        h, w = frame.shape[:2]
        
        # Counting lines (same as original)
        count_line_horizontal = int(0.5 * h)
        count_line_vertical = int(0.7 * w)
        count_zone_start = int(0.45 * h)
        count_zone_end = int(0.55 * h)
        count_zone_top = int(0.3 * h)
        count_zone_bottom = int(0.7 * h)
        vertical_zone_left = int(0.65 * w)
        vertical_zone_right = int(0.75 * w)
        roi_margin = int(w * 0.15)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # KEEP ORIGINAL SENSITIVE SETTINGS
        bg = cv2.createBackgroundSubtractorMOG2(
            history=150,
            varThreshold=12,
            detectShadows=False
        )
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        
        tracks = []
        next_id = 0
        vehicle_count = 0
        frame_count = 0
        counted_ids = set()
        recent_counts = []
        max_recent = 5
        last_count_frame = -100
        
        # Track statistics
        rejected_stationary = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            display = frame.copy()
            
            # Background subtraction 
            fg = bg.apply(frame, learningRate=-1)
            
            # Morphology 
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=2)
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=4)
            fg = cv2.dilate(fg, kernel, iterations=1)

            movement_ratio = np.count_nonzero(fg) / (fg.shape[0] * fg.shape[1])
            if movement_ratio < 0.0005: 
                if debug:
                    cv2.putText(frame, "STATUS: IDLE (No Movement)", (20, 100), 0, 1, (0, 0, 255), 2)
                    cv2.imshow("Stability Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                continue
            
            # contours
            contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            #Extract detections 
            detections = []
            raw_detections = []
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                x, y, wc, hc = cv2.boundingRect(cnt)
                cx = x + wc/2
                cy = y + hc/2
                
                raw_detections.append((cx, cy, (x, y, wc, hc), area))
                
                is_edge = (cx < w * 0.15 or cx > w * 0.85)
                min_area = 600 if is_edge else 800
                
                if area < min_area:
                    continue
                
                if y < h * 0.10 or y > h * 0.90:
                    continue
                
                if cx < w * 0.08 or cx > w * 0.92:
                    continue
                
                aspect = wc / hc if hc > 0 else 0
                if aspect > 3.5:
                    continue
                
                if hc < 25:
                    continue
                
                # Only reject OBVIOUS person shapes
                if check_person_like((x, y, wc, hc), frame.shape):
                    continue
                
                detections.append((cx, cy, (x, y, wc, hc)))
            
            # STEP 5: Fusion 
            detections = fuse_detections(detections, x_thresh=150, y_thresh=120)
            
            # STEP 6: Tracking
            predicted = []
            for tr in tracks:
                px, py = tr.kf.predict()
                predicted.append((px, py))
                tr.age += 1
                
                # Update motion metrics
                total_disp, recent_move, is_moving = calculate_motion_metrics(tr)
                tr.max_displacement = max(tr.max_displacement, total_disp)
                
                # Track stationary frames (soft)
                if recent_move < 1.0:
                    tr.stationary_count += 1
                else:
                    tr.stationary_count = 0
                    tr.motion_score += recent_move
            
            # Cost matrix
            cost = np.zeros((len(predicted), len(detections)), dtype=np.float32)
            for i, (px, py) in enumerate(predicted):
                for j, (cx, cy, _) in enumerate(detections):
                    cost[i, j] = np.hypot(px-cx, py-cy)
            
            matched = []
            if cost.size > 0:
                r, c = linear_sum_assignment(cost)
                for i, j in zip(r, c):
                    if cost[i, j] < 120:
                        matched.append((i, j))
            
            used_tracks, used_dets = set(), set()
            
            # Update matched tracks
            for i, j in matched:
                tr = tracks[i]
                cx, cy, box = detections[j]
                tr.kf.update((cx, cy))
                tr.prev_y = tr.centroid[1]
                tr.prev_x = tr.centroid[0]
                tr.centroid = (cx, cy)
                tr.box = box
                tr.missed = 0
                tr.positions.append((cx, cy))
                if len(tr.positions) > 20:
                    tr.positions.pop(0)
                
                # Direction detection
                if len(tr.positions) >= 5:
                    x_movement = tr.positions[-1][0] - tr.positions[0][0]
                    y_movement = tr.positions[-1][1] - tr.positions[0][1]
                    
                    if abs(x_movement) > abs(y_movement) * 1.5:
                        tr.direction = 'turning'
                    else:
                        tr.direction = 'straight'
                
                used_tracks.add(i)
                used_dets.add(j)
            
            # Increment missed
            for i, tr in enumerate(tracks):
                if i not in used_tracks:
                    tr.missed += 1
            
            # Remove lost tracks
            tracks = [t for t in tracks if t.missed < 15]
            
            # Create new tracks
            for j, (cx, cy, box) in enumerate(detections):
                if j not in used_dets:
                    new_track = Track((cx, cy), box, next_id)
                    tracks.append(new_track)
                    next_id += 1
            
            # Track deduplication (SAME AS ORIGINAL)
            tracks_to_remove = set()
            
            for i in range(len(tracks)):
                if i in tracks_to_remove:
                    continue
                    
                for j in range(i+1, len(tracks)):
                    if j in tracks_to_remove:
                        continue
                    
                    ti, tj = tracks[i], tracks[j]
                    
                    h_dist = abs(ti.centroid[0] - tj.centroid[0])
                    v_dist = abs(ti.centroid[1] - tj.centroid[1])
                    
                    if len(ti.positions) >= 2 and len(tj.positions) >= 2:
                        ti_vy = ti.positions[-1][1] - ti.positions[0][1]
                        tj_vy = tj.positions[-1][1] - tj.positions[0][1]
                        vel_diff = abs(ti_vy - tj_vy)
                    else:
                        vel_diff = abs(ti.kf.x[3] - tj.kf.x[3])
                    
                    is_duplicate = False
                    
                    if h_dist < 120 and v_dist < 100:
                        is_duplicate = True
                    
                    if h_dist < 180 and vel_diff < 3.0 and abs(v_dist) < 120:
                        is_duplicate = True
                    
                    if h_dist < 150 and v_dist < 100 and abs(ti.age - tj.age) < 10:
                        is_duplicate = True
                    
                    if is_duplicate:
                        if ti.age > tj.age or (ti.age == tj.age and ti.id < tj.id):
                            tracks_to_remove.add(j)
                            if tj.id in counted_ids:
                                counted_ids.add(ti.id)
                                ti.counted = True
                        else:
                            tracks_to_remove.add(i)
                            if ti.id in counted_ids:
                                counted_ids.add(tj.id)
                                tj.counted = True
                        break
            
            tracks = [t for i, t in enumerate(tracks) if i not in tracks_to_remove]
            
            # COUNTING LOGIC - WITH SOFT MOTION CHECK
            for tr in tracks:
                if tr.id in counted_ids:
                    continue
                
                if tr.age < 1:
                    continue
                
                cy = tr.centroid[1]
                py = tr.prev_y
                cx = tr.centroid[0]
                px = tr.prev_x
                
                is_clearly_stationary = (
                    tr.stationary_count > 25 and  # Stationary for 25+ frames
                    tr.max_displacement < 8 and    # Never moved much
                    tr.age > 30                     # Been around a while
                )
                
                if is_clearly_stationary:
                    if debug and tr.id not in rejected_stationary:
                        rejected_stationary.append(tr.id)
                        print(f"⊗ Rejected ID:{tr.id} (stationary: disp={tr.max_displacement:.1f}, static={tr.stationary_count})")
                    continue
                
                # Check nearby counted tracks
                nearby_counted = False
                for other_tr in tracks:
                    if other_tr.id == tr.id or other_tr.id not in counted_ids:
                        continue
                    if abs(other_tr.centroid[0] - tr.centroid[0]) < 150 and \
                       abs(other_tr.centroid[1] - tr.centroid[1]) < 120:
                        nearby_counted = True
                        break
                
                if nearby_counted:
                    continue
                
                # Line crossing logic 
                crossed_horizontal = False
                if not tr.crossed_horizontal:
                    if (py < count_line_horizontal and cy >= count_line_horizontal) or \
                       (py > count_line_horizontal and cy <= count_line_horizontal):
                        crossed_horizontal = True
                        tr.crossed_horizontal = True
                
                crossed_vertical = False
                if not tr.crossed_vertical:
                    if (px < count_line_vertical and cx >= count_line_vertical) or \
                       (px > count_line_vertical and cx <= count_line_vertical):
                        if cy > h * 0.3:
                            crossed_vertical = True
                            tr.crossed_vertical = True
                
                in_zone = (count_zone_start <= cy <= count_zone_end)
                was_outside = (py < count_zone_start or py > count_zone_end)
                entered_zone = in_zone and was_outside
                
                in_vertical_zone = (vertical_zone_left <= cx <= vertical_zone_right)
                was_left = (px < vertical_zone_left)
                entered_vertical_zone = in_vertical_zone and was_left
                
                has_movement = False
                crossed_line_in_history = False
                
                if len(tr.positions) >= 3:
                    first_y = tr.positions[0][1]
                    last_y = tr.positions[-1][1]
                    first_x = tr.positions[0][0]
                    last_x = tr.positions[-1][0]
                    
                    y_displacement = abs(last_y - first_y)
                    x_displacement = abs(last_x - first_x)
                    
                    if y_displacement > 10 or x_displacement > 10:
                        has_movement = True
                        
                        for i in range(1, len(tr.positions)):
                            prev_pos_y = tr.positions[i-1][1]
                            curr_pos_y = tr.positions[i][1]
                            prev_pos_x = tr.positions[i-1][0]
                            curr_pos_x = tr.positions[i][0]
                            
                            if not tr.crossed_horizontal:
                                if (prev_pos_y < count_line_horizontal and curr_pos_y >= count_line_horizontal) or \
                                   (prev_pos_y > count_line_horizontal and curr_pos_y <= count_line_horizontal):
                                    crossed_line_in_history = True
                                    tr.crossed_horizontal = True
                                    break
                            
                            if not tr.crossed_vertical and curr_pos_y > h * 0.3:
                                if (prev_pos_x < count_line_vertical and curr_pos_x >= count_line_vertical) or \
                                   (prev_pos_x > count_line_vertical and curr_pos_x <= count_line_vertical):
                                    crossed_line_in_history = True
                                    tr.crossed_vertical = True
                                    break
                
                in_expanded_zone = (count_zone_top <= cy <= count_zone_bottom)
                if in_expanded_zone and has_movement and tr.age >= 3 and not tr.crossed_horizontal:
                    crossed_horizontal = True
                    tr.crossed_horizontal = True
                
                # Counting decision
                should_count = False
                count_reason = ""
                
                if crossed_horizontal and not tr.crossed_vertical:
                    should_count = True
                    count_reason = "horizontal"
                elif crossed_vertical and not tr.crossed_horizontal:
                    should_count = True
                    count_reason = "vertical"
                elif crossed_horizontal and tr.crossed_vertical:
                    if not tr.counted:
                        should_count = True
                        count_reason = "first-cross"
                elif entered_zone or entered_vertical_zone or crossed_line_in_history:
                    if not tr.crossed_horizontal and not tr.crossed_vertical:
                        should_count = True
                        count_reason = "zone-entry"
                
                if should_count:
                    vehicle_count += 1
                    counted_ids.add(tr.id)
                    tr.counted = True
                    last_count_frame = frame_count
                    
                    recent_counts.append((frame_count, tr.id, count_reason))
                    if len(recent_counts) > max_recent:
                        recent_counts.pop(0)
                    
                    if debug:
                        direction_str = tr.direction or "unknown"
                        print(f"✓ Counted ID:{tr.id} F:{frame_count} (disp={tr.max_displacement:.1f}, motion={tr.motion_score:.1f}, dir={direction_str}, reason={count_reason})")
                    
                    for other_tr in tracks:
                        if other_tr.id == tr.id:
                            continue
                        if abs(other_tr.centroid[0] - tr.centroid[0]) < 120 and \
                           abs(other_tr.centroid[1] - tr.centroid[1]) < 100:
                            counted_ids.add(other_tr.id)
                            other_tr.counted = True
                            if debug:
                                print(f"  → Marking ID:{other_tr.id} as counted (duplicate)")
            
            # VISUALIZATION
            if debug:
                display_clean = frame.copy()
                
                cv2.rectangle(display_clean, (0, count_zone_start), (w, count_zone_end), 
                            (0, 255, 255), 2)
                cv2.line(display_clean, (0, count_line_horizontal), (w, count_line_horizontal), 
                        (0, 255, 255), 3)
                cv2.line(display_clean, (count_line_vertical, 0), (count_line_vertical, h), 
                        (255, 255, 0), 3)
                cv2.rectangle(display_clean, (vertical_zone_left, int(h*0.3)), 
                            (vertical_zone_right, h), (255, 255, 0), 2)
                cv2.line(display_clean, (roi_margin, 0), (roi_margin, h), (255, 0, 255), 1)
                cv2.line(display_clean, (w-roi_margin, 0), (w-roi_margin, h), (255, 0, 255), 1)
                
                for (cx, cy, (x, y, wc, hc), area) in raw_detections:
                    cv2.rectangle(display_clean, (x, y), (x+wc, y+hc), (0, 165, 255), 1)
                
                for (cx, cy, (x, y, wc, hc)) in detections:
                    cv2.rectangle(display_clean, (x, y), (x+wc, y+hc), (0, 255, 255), 2)
                    cv2.circle(display_clean, (int(cx), int(cy)), 3, (0, 255, 255), -1)
                
                for tr in tracks:
                    x, y, wc, hc = tr.box
                    cx, cy = tr.centroid
                    
                    is_stat = (tr.stationary_count > 10 and tr.max_displacement < 10)
                    
                    if tr.id in counted_ids:
                        color = (255, 0, 0)  # Blue - counted
                    elif is_stat:
                        color = (128, 0, 128)  # Purple - stationary
                    elif tr.missed > 0:
                        color = (0, 0, 255)  # Red - losing
                    else:
                        color = (0, 255, 0)  # Green - active
                    
                    cv2.rectangle(display_clean, (x, y), (x+wc, y+hc), color, 3)
                    
                    for i in range(1, len(tr.positions)):
                        pt1 = (int(tr.positions[i-1][0]), int(tr.positions[i-1][1]))
                        pt2 = (int(tr.positions[i][0]), int(tr.positions[i][1]))
                        cv2.line(display_clean, pt1, pt2, color, 2)
                    
                    cv2.circle(display_clean, (int(cx), int(cy)), 5, color, -1)
                    
                    label = f"{tr.id}"
                    if is_stat:
                        label += "*"
                    cv2.putText(display_clean, label, (x, y-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                fig, axes = plt.subplots(2, 2, figsize=(16, 10), 
                                        gridspec_kw={'height_ratios': [1, 4]})
                
                axes[0, 0].set_xlim(0, 1)
                axes[0, 0].set_ylim(0, 1)
                axes[0, 0].axis('off')
                
                frames_since_count = frame_count - last_count_frame
                if frames_since_count < 10:
                    bg_color = 'lime' if frames_since_count < 3 else 'green'
                    text_color = 'black'
                    flash_text = " NEW! " if frames_since_count < 3 else ""
                else:
                    bg_color = 'black'
                    text_color = 'green'
                    flash_text = ""
                
                axes[0, 0].text(0.5, 0.7, f"COUNT: {vehicle_count}{flash_text}", 
                              ha='center', va='center', fontsize=60, 
                              fontweight='bold', color=text_color,
                              bbox=dict(boxstyle='round', facecolor=bg_color, alpha=0.9, pad=0.8))
                
                axes[0, 0].text(0.5, 0.3, f"Frame: {frame_count} | Tracks: {len(tracks)} | Rejected: {len(rejected_stationary)}", 
                              ha='center', va='center', fontsize=14, color='white',
                              bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
                
                axes[0, 1].set_xlim(0, 1)
                axes[0, 1].set_ylim(0, 1)
                axes[0, 1].axis('off')
                
                axes[0, 1].text(0.5, 0.98, 'COUNTING MONITOR', 
                              ha='center', va='top', fontsize=16, 
                              fontweight='bold', color='yellow',
                              bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
                
                if recent_counts:
                    y_pos = 0.85
                    axes[0, 1].text(0.05, y_pos, "Recent Counts:", 
                                  ha='left', va='center', fontsize=11, 
                                  color='white', fontweight='bold')
                    y_pos -= 0.12
                    
                    for i, (frame_num, vid, reason) in enumerate(reversed(recent_counts)):
                        age = frame_count - frame_num
                        alpha = max(0.4, 1.0 - (age / 100))
                        color = 'lime' if age < 10 else 'yellow' if age < 30 else 'gray'
                        
                        text = f"  ✓ ID:{vid} F{frame_num} [{reason}]"
                        axes[0, 1].text(0.05, y_pos, text, 
                                      ha='left', va='center', fontsize=10, 
                                      color=color, alpha=alpha,
                                      family='monospace')
                        y_pos -= 0.10
                        if y_pos < 0.3:
                            break
                else:
                    axes[0, 1].text(0.5, 0.7, "Waiting for vehicles...", 
                                  ha='center', va='center', fontsize=11, 
                                  color='gray', style='italic')
                
                y_pos = 0.25
                active = len([t for t in tracks if t.id not in counted_ids])
                axes[0, 1].text(0.05, y_pos, f"Active: {active} | Counted: {len(counted_ids)}", 
                              ha='left', va='center', fontsize=10, 
                              color='cyan', fontweight='bold')
                y_pos -= 0.10
                
                for i, tr in enumerate(tracks[:3]):
                    status = "COUNTED" if tr.id in counted_ids else "TRACK"
                    color_code = 'blue' if status == "COUNTED" else 'lime'
                    
                    cross_info = ""
                    if tr.crossed_horizontal:
                        cross_info += "H"
                    if tr.crossed_vertical:
                        cross_info += "V"
                    if cross_info:
                        cross_info = f"[{cross_info}]"
                    
                    direction_marker = ""
                    if tr.direction == 'straight':
                        direction_marker = "→"
                    elif tr.direction == 'turning':
                        direction_marker = "↱"
                    
                    text = f"  ID:{tr.id} {status} Disp:{tr.max_displacement:.0f} Stat:{tr.stationary_count} {direction_marker}{cross_info}"
                    axes[0, 1].text(0.05, y_pos, text, 
                                  ha='left', va='center', fontsize=8, 
                                  color=color_code, family='monospace')
                    y_pos -= 0.08
                
                axes[0, 1].text(0.05, 0.05, f"Raw: {len(raw_detections)} | Fused: {len(detections)} | Total: {vehicle_count}", 
                              ha='left', va='center', fontsize=9, 
                              color='white', family='monospace',
                              bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
                
                rgb = cv2.cvtColor(display_clean, cv2.COLOR_BGR2RGB)
                axes[1, 0].imshow(rgb)
                axes[1, 0].set_title('Video Feed (Clean)', 
                                   fontsize=14, fontweight='bold', color='white',
                                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
                axes[1, 0].axis('off')
                
                axes[1, 1].imshow(fg, cmap='gray')
                axes[1, 1].set_title('Foreground Mask', fontsize=14, fontweight='bold')
                axes[1, 1].axis('off')
                
                legend_text = "Orange=Raw | Yellow=Fused | Green=Track | Blue=Counted | Purple=Stationary* | Cyan=H-Line | Yellow=V-Line"
                fig.text(0.27, 0.01, legend_text, ha='center', fontsize=9, 
                        color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
                
                plt.tight_layout()
                plt.show()
                clear_output(wait=True)
                time.sleep(0.01)
        
        cap.release()
        print(f"\nTotal Frames Processed: {frame_count}")
        print(f"Unique IDs Created: {next_id}")
        print(f"Rejected Stationary: {len(rejected_stationary)}")
        return vehicle_count


# =========================
# USAGE
# =========================
if __name__ == "__main__":
    video_path = "vehant_hackathon_video_1.avi"
    solver = Solution()
    count = solver.forward(video_path, debug=False)
    print(f"VEHICLE COUNT: {count}")