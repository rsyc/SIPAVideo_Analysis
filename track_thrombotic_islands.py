import cv2
import numpy as np
import pandas as pd
from collections import defaultdict
import os
from datetime import timedelta
from pathlib import Path


# ====================== USER TUNABLE PARAMETERS ======================
VIDEO_PATH = "C:/Users/rojan/Documents/FSU/codes/PythonCodes/SIPAVideos/REACT004Visit5_March25_Straight1_Channel5_Exercise_NaHeparin_narrowTubing_height15cm_141ulMediumCBD_FIXED.mp4"
# "C:/Users/rojan/Documents/FSU/codes/PythonCodes/SIPAVideos/REACT009Visit3_April2_Straight1_Channel6_Exercise_NaHeparin_narrowTubing_height15cm_141ulMediumCBD_FIXED.mp4"          # <--- CHANGE THIS
# "C:/Users/rojan/Documents/FSU/codes/PythonCodes/SIPAVideos/REACT009Visit3_April2_Straight1_Channel5_REST_NaHeparin_narrowTubing_height15cm_141ulMediumCBD.avi"
# "C:/Users/rojan/Documents/FSU/codes/PythonCodes/SIPAVideos/REACT009Visit3_April2_Straight1_Channel6_Exercise_NaHeparin_narrowTubing_height15cm_141ulMediumCBD.avi"
# "C:/Users/rojan/Documents/FSU/codes/PythonCodes/SIPAVideos/REACT004Visit5_March25_Straight1_Channel5_Exercise_NaHeparin_narrowTubing_height15cm_141ulMediumCBD.avi"
base_name = Path(VIDEO_PATH).stem

BRIGHTNESS_THRESH = 110                # increase if too many false positives
DIFF_THRESHOLD = 5 #7  #25             # NEW tuning knob: lower = more sensitive, higher = stricter to noise

MIN_AREA = 30                          # pixels — ignore tiny noise
MAX_DISTANCE = 45                      # pixels — how far an island can "move" between frames
MAX_DISAPPEARED = 8                    # frames an island can vanish before we forget it
BLUR_KERNEL = (5, 5)                   # gentle smoothing
WRITE_OUTPUT_VIDEO = True              # Set False to skip writing video → huge speed boost!


OUTPUT_VIDEO = base_name + "_Annotated_10frames_Mp4.avi"
# "REACT009Visit3_April2_Straight1_Channel5_REST_NaHeparin_narrowTubing_height15cm_141ulMediumCBD_Annotated_10frames.avi"
OUTPUT_CSV = base_name + "_islands_10frames_Mp4.csv"
# "REACT009Visit3_April2_Straight1_Channel5_REST_NaHeparin_narrowTubing_height15cm_141ulMediumCBD_islands_10frames.csv"
frame_num = 200 #1500 #4000   # to make sure that we start with a frame that blood is fully developed and coveres the channel
NUM_BACKGROUND_FRAMES = 1800 #1500           # frames used to build the static background model
                                       # → set this BEFORE any islands appear
                                       
# NEW: Robust reading controls
REOPEN_EVERY_N_FRAMES = 8000           # re-open capture periodically (prevents index crash)
MAX_RECOVERY_RETRIES = 100

# === SENSITIVITY IMPROVEMENT (new) ===
DILATION_ITERATIONS = 5                # 1 = slight expansion, 2-3 = captures more halo, 0 = original behavior
DILATION_KERNEL_SIZE = 3               # usually 3 or 5
                         
# =====================================================================
class CentroidTracker:
    def __init__(self, maxDisappeared=MAX_DISAPPEARED, maxDistance=MAX_DISTANCE):
        self.nextObjectID = 0
        self.objects = {}          # id → (centroid, area, brightness)
        self.disappeared = {}
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance

    def register(self, centroid, area, brightness):
        self.objects[self.nextObjectID] = (centroid, area, brightness)
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, detections):
        # detections = list of (centroid, area, brightness)
        if len(detections) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.maxDisappeared:
                    self.deregister(oid)
            return self.objects

        inputCentroids = [d[0] for d in detections]
        inputAreas = [d[1] for d in detections]
        inputBright = [d[2] for d in detections]

        if len(self.objects) == 0:
            for i in range(len(inputCentroids)):
                self.register(inputCentroids[i], inputAreas[i], inputBright[i])
            return self.objects

        objectIDs = list(self.objects.keys())
        objectCentroids = [self.objects[oid][0] for oid in objectIDs]

        # distance matrix
        D = np.zeros((len(objectIDs), len(inputCentroids)))
        for i, oc in enumerate(objectCentroids):
            for j, ic in enumerate(inputCentroids):
                D[i, j] = np.sqrt((oc[0] - ic[0]) ** 2 + (oc[1] - ic[1]) ** 2)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        usedRows = set()
        usedCols = set()

        for row, col in zip(rows, cols):
            if row in usedRows or col in usedCols:
                continue
            if D[row, col] > self.maxDistance:
                continue
            objectID = objectIDs[row]
            self.objects[objectID] = (inputCentroids[col], inputAreas[col], inputBright[col])
            self.disappeared[objectID] = 0
            usedRows.add(row)
            usedCols.add(col)

        # deregister lost objects
        for row in set(range(len(objectIDs))) - usedRows:
            oid = objectIDs[row]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.maxDisappeared:
                self.deregister(oid)

        # register new islands
        for col in set(range(len(inputCentroids))) - usedCols:
            self.register(inputCentroids[col], inputAreas[col], inputBright[col])

        return self.objects

# ======================= draw and rotate ROI BOX ================
def select_rotated_movable_roi(first_frame):
    """Improved: Draw box → Rotate + Move with live preview → ESC to confirm"""
    print("Step 1: Draw initial axis-aligned box around the channel")
    roi = cv2.selectROI("Draw initial box", first_frame, False)
    cv2.destroyAllWindows()

    x, y, w, h = map(int, roi)
    if w == 0 or h == 0:
        print("No box selected → using full frame")
        return 0, 0, first_frame.shape[1], first_frame.shape[0], 0.0, None

    win_name = "Adjust ROI: Rotate + Move (ESC = Confirm)"
    cv2.namedWindow(win_name)

    cv2.createTrackbar("Angle (-45 to +45)", win_name, 45, 90, lambda v: None)
    cv2.createTrackbar("Move X", win_name, x + w//2, first_frame.shape[1], lambda v: None)
    cv2.createTrackbar("Move Y", win_name, y + h//2, first_frame.shape[0], lambda v: None)

    def redraw():
        angle = cv2.getTrackbarPos("Angle (-45 to +45)", win_name) - 45
        cx = cv2.getTrackbarPos("Move X", win_name)
        cy = cv2.getTrackbarPos("Move Y", win_name)

        # Rotation matrix around current center
        rot_mat = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

        # Original corners relative to initial box
        pts = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.float32)
        rotated_pts = cv2.transform(np.array([pts]), rot_mat)[0].astype(np.int32)

        overlay = first_frame.copy()
        cv2.polylines(overlay, [rotated_pts], True, (0, 255, 0), 4)
        cv2.circle(overlay, (cx, cy), 8, (0, 0, 255), -1)   # red center marker

        cv2.putText(overlay, f"Angle: {angle:.1f}°", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(overlay, f"Center: ({cx}, {cy})", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(overlay, "ESC = Confirm when green box matches channel", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        cv2.imshow(win_name, overlay)

    print("Adjust using the 3 trackbars until the green box perfectly fits the channel")
    while True:
        redraw()
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    final_angle = cv2.getTrackbarPos("Angle (-45 to +45)", win_name) - 45
    final_cx = cv2.getTrackbarPos("Move X", win_name)
    final_cy = cv2.getTrackbarPos("Move Y", win_name)
    cv2.destroyAllWindows()

    # === Calculate tight final bounding box that contains the rotated region ===
    rot_mat = cv2.getRotationMatrix2D((final_cx, final_cy), final_angle, 1.0)
    pts = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.float32)
    rotated_pts = cv2.transform(np.array([pts]), rot_mat)[0]

    min_x = int(rotated_pts[:, 0].min())
    min_y = int(rotated_pts[:, 1].min())
    max_x = int(rotated_pts[:, 0].max())
    max_y = int(rotated_pts[:, 1].max())

    final_w = max_x - min_x
    final_h = max_y - min_y
    final_x = min_x
    final_y = min_y

    # Clamp to image boundaries
    final_x = max(0, final_x)
    final_y = max(0, final_y)
    final_w = min(final_w, first_frame.shape[1] - final_x)
    final_h = min(final_h, first_frame.shape[0] - final_y)

    print(f"Final ROI: x={final_x}, y={final_y}, w={final_w}, h={final_h}, angle={final_angle:.1f}°")

    # Create accurate rotated mask
    mask = np.zeros((first_frame.shape[0], first_frame.shape[1]), dtype=np.uint8)
    cv2.fillPoly(mask, [rotated_pts.astype(np.int32)], 255)

    # Optional: slight dilation on mask for safety
    dilate_kernel = np.ones((5, 5), np.uint8)
    rotated_mask = cv2.dilate(mask, dilate_kernel, iterations=1)

    return final_x, final_y, final_w, final_h, final_angle, rotated_mask


# ====================== ROBUST VIDEO READER ======================
def create_cap():
    """Create VideoCapture with FFMPEG backend (best for long AVIs)"""
    return cv2.VideoCapture(VIDEO_PATH, cv2.CAP_FFMPEG)

# ====================== MAIN SCRIPT ======================
cap = create_cap()              
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# Set position to the very end (after the last frame)
cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames)
# Get the timestamp of the last frame in milliseconds
duration_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
duration_sec = duration_ms / 1000.0
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # total_frames/duration_sec +1 #cap.get(cv2.CAP_PROP_FPS) or 30.0
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video loaded — {total_frames} frames @ {fps} fps")

# ====================== INTERACTIVE ROI SELECTION ======================
cap.set(cv2.CAP_PROP_POS_FRAMES, NUM_BACKGROUND_FRAMES)
ret, first_frame = cap.read()
if not ret:
    raise RuntimeError("Could not read first frame")

# print("Please draw a box around the stenotic channel (the area where islands appear)")
# roi = cv2.selectROI("Select Stenotic Channel ROI — drag & press ENTER", first_frame, False)
# cv2.destroyAllWindows()

# roi_x, roi_y, roi_w, roi_h = map(int, roi)
roi_x, roi_y, roi_w, roi_h, rotation_angle, rotated_mask  = select_rotated_movable_roi(first_frame)

if roi_w == 0 or roi_h == 0:
    print("No ROI selected — using full frame")
    roi_x = roi_y = 0
    roi_w, roi_h = frame_width, frame_height

# === CRITICAL: Clamp to avoid empty crops and negative indices ===
roi_x = max(0, roi_x)
roi_y = max(0, roi_y)
roi_w = min(roi_w, first_frame.shape[1] - roi_x)
roi_h = min(roi_h, first_frame.shape[0] - roi_y)

print(f"ROI selected: x={roi_x}, y={roi_y}, w={roi_w}, h={roi_h}, angle={rotation_angle:.1f}° (only this area will be analyzed)")

# ====================== INTERACTIVE SCALE BAR CALIBRATION ======================
print("Now calibrate the scale bar")
print("Click TWO points on the scale-bar line (anywhere on the line), then press ESC")

# Create a fresh copy of the first frame for calibration
scale_frame = first_frame.copy()

# Make the window resizable + normal (this helps with taskbar issues)
cv2.namedWindow("Scale Bar Calibration", cv2.WINDOW_NORMAL)

# Resize to a comfortable size that should fit on most screens
# You can adjust these numbers if needed (e.g. 1200, 800)
cv2.resizeWindow("Scale Bar Calibration", 1440, 960)

# Move the window to the top-left so it doesn't hide under the taskbar
cv2.moveWindow("Scale Bar Calibration", 50, 50)

scale_points = []
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        scale_points.append((x, y))
        cv2.circle(scale_frame, (x, y), 8, (0, 255, 255), -1)
        if len(scale_points) == 2:
            cv2.line(scale_frame, scale_points[0], scale_points[1], (0, 255, 255), 3)
        cv2.imshow("Scale Bar Calibration", scale_frame)

cv2.setMouseCallback("Scale Bar Calibration", mouse_callback)
cv2.imshow("Scale Bar Calibration", scale_frame)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or len(scale_points) == 2:   # ESC key
        break

cv2.destroyAllWindows()

if len(scale_points) != 2:
    print("Scale calibration skipped — areas will stay in pixels")
    scale_um_per_px = 1.0
else:
    pixel_length = np.hypot(scale_points[1][0] - scale_points[0][0],
                            scale_points[1][1] - scale_points[0][1])
    try:
        scale_um = float(input("\nEnter the real length of the scale bar (in µm): "))
        if scale_um > 500:
            scale_um = scale_um/1.5  # The microscope is usually set to 1.5* zoom but 
                                    # it is not applied to the annotation. The actual width of the channel is 
                                    # about 480um but the annotation usually shows about 700um 
        scale_um_per_px = scale_um / pixel_length
        print(f"Scale calibrated: {scale_um:.2f} µm = {pixel_length:.1f} px → {scale_um_per_px:.4f} µm/pixel")
    except ValueError:
        print("Invalid number entered. Using pixel units.")
        scale_um_per_px = 1.0

area_scale_factor = scale_um_per_px ** 2   # for area conversion

# ====================== BUILD BACKGROUND MODEL (median of first N frames) ======================
print(f"Building background model from first {NUM_BACKGROUND_FRAMES} frames...")
bg_frames = []
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
for i in range(NUM_BACKGROUND_FRAMES):
    ret, frame = cap.read()
    # cv2.imshow('frame', frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    if not ret:
        print(f" {i}: Video ended before background could be built — using fewer frames")
        continue
    roi_frame = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    if len(roi_frame.shape) == 3:
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi_frame.copy()
    # bg_frames.append(gray)
    # Apply the rotated mask (only keep pixels inside your green box)
    local_mask = rotated_mask[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    gray_masked = cv2.bitwise_and(gray, gray, mask=local_mask)

    bg_frames.append(gray_masked.astype(np.float32))

if len(bg_frames) == 0:
    raise RuntimeError("Could not build background")

bg = np.median(bg_frames, axis=0).astype(np.uint8)
# cv2.imshow('bg', bg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
print("Background model ready — static bright regions will now be ignored")



# ====================== OUTPUT VIDEO (optional) ======================
if WRITE_OUTPUT_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')   # or 'avc1'    # cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (frame_width, frame_height))
else:
    print("WRITE_OUTPUT_VIDEO = False → skipping video write (much faster!)")

tracker = CentroidTracker()
data = defaultdict(list)   # island_id → list of (time_sec, area, brightness, cx, cy)

# Reset video to the frame we want it to start so we analyze the entire video
# cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
kernel =  cv2.getStructuringElement(cv2.MORPH_CROSS,(7,7)) 
         #cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)) 
         #np.ones((3, 3), np.uint8)   # for morphology
recovery_attempts = 0
# last_successful_frame = frame_num # 0
last_objects = {}          # hold the last known islands for skipped frames

while cap.isOpened():
    # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print(f"Read failed at frame {frame_num}. Attempting recovery...")
        cap.release()

        if recovery_attempts >= MAX_RECOVERY_RETRIES:
            print("Max recovery attempts reached. Stopping.")
            break

        # Re-open and try to seek to where we left off
        cap = create_cap()
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num) #frame_num)   # jump to the next expected frame
        recovery_attempts += 1
        # Try one more read immediately
        ret, frame = cap.read()
        if not ret:
            print("Still failing after reopen — stopping.")
            break
    
    # Check the frame
    # cv2.imshow('Frame', frame)
    # cv2.waitKey(0)
    frame_num += 1

    # Periodic reopen (prevents the 2300-frame crash)
    # if frame_num % REOPEN_EVERY_N_FRAMES == 0:
    #     print(f"Periodic reopen at frame {frame_num} (to keep index stable)")
    #     cap.release()
    #     cap = create_cap()
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    #     continue   # skip processing this frame (we just reopened)
        
    time_sec = frame_num / fps
    # # ====================== SKIP PROCESSING (but still write the frame) ======================
    # if frame_num % 2 != 0:
    #     # Write the original frame to keep full video length & timing
    #     if WRITE_OUTPUT_VIDEO:
    #         # Draw ROI box + last-known islands (no new detection)
    #         cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 100, 0), 2)
    #         cv2.putText(frame, "ROI", (roi_x + 10, roi_y - 10),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
    #         for oid, (centroid, area_px, brightness) in last_objects.items():
    #             cx, cy = centroid
    #             area_um2 = area_px * area_scale_factor
    #             # cv2.putText(...)   # comment out if you don't want text
    #             cv2.circle(frame, (cx, cy), 4, (0, 255, 200), -1)
    #         out.write(frame)
    #     continue
    # ====================== PROCESS EVERY 100th FRAME ======================
    # Crop to ROI for detection (this is the speed boost!)
    roi_frame = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    
    # convert to grayscale
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(roi_frame , cv2.COLOR_BGR2GRAY)
    else:
        gray = roi_frame .copy()
    # cv2.imshow('gray', gray)
    # cv2.waitKey(0)
    
    # Apply the rotated mask (only keep pixels inside the green rotated box)
    # We need to create a local mask for this crop
    local_mask = rotated_mask[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    gray = cv2.bitwise_and(gray, gray, mask=local_mask)

    # preprocess
    # Background subtraction
    diff = cv2.subtract(gray, bg)   # note: bg is masked before  # cv2.absdiff(bg, gray)    # or cv2.subtract(gray, bg) if you prefer only brighter
    blurred = cv2.GaussianBlur(diff, BLUR_KERNEL, 0)
    # blurred = cv2.GaussianBlur(gray, BLUR_KERNEL, 0)
    # cv2.imshow('blurred', blurred)
    # cv2.waitKey(0)
    
    # Fixed threshold (you can lower DIFF_THRESHOLD a bit if needed)
    _, thresh = cv2.threshold(blurred, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    # _, thresh = cv2.threshold(blurred, BRIGHTNESS_THRESH, 255, cv2.THRESH_BINARY)

    # Morphology to clean noise
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # === EXPAND TO CAPTURE FULL ISLAND HALO ===
    dilate_kernel = np.ones((DILATION_KERNEL_SIZE, DILATION_KERNEL_SIZE), np.uint8)
    thresh = cv2.dilate(thresh, dilate_kernel, iterations=DILATION_ITERATIONS)
    
    # Find contours on the difference image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Plot contours on the original frame (for checking)
    # Parameters: (target_image, contours_list, contour_index, color, thickness)
    # Use -1 as contour_index to draw ALL contours
    # cv2.drawContours(roi_frame , contours, -1, (0, 255, 0), 3)    
    # cv2.imshow('Contours', frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()   
    
    detections = []
    for cnt in contours:
        area_px  = cv2.contourArea(cnt)
        if area_px  < MIN_AREA:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # mean brightness from ORIGINAL gray (not the diff)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        brightness = cv2.mean(gray, mask=mask)[0]

        # back to full-frame coordinates
        full_cx = cx + roi_x
        full_cy = cy + roi_y

        detections.append(((full_cx, full_cy), area_px , brightness))

        # draw on output frame (for visual check)
        cv2.drawContours(roi_frame, [cnt], -1, (0, 255, 0), 2)
        
        # optional: draw raw contour on roi for debugging (comment out if not needed)
        # cv2.drawContours(roi_frame, [cnt], -1, (0, 255, 0), 1)

    # update tracker
    objects = tracker.update(detections)
    last_objects = objects.copy()   # save for skipped frames

    # record data
    for oid, (centroid, area_px , brightness) in objects.items():
        cx, cy = centroid
        area_um2 = area_px * area_scale_factor
        data[oid].append({
            "time_sec": time_sec,
            "area_px": area_px,
            "area_um2": round(area_um2, 3),
            "mean_brightness": round(brightness, 2),
            "centroid_x": cx,
            "centroid_y": cy
        })
        
    # ====================== DRAW ON OUTPUT FRAME ======================
    if WRITE_OUTPUT_VIDEO:
        # Draw ROI box
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 100, 0), 2)
        cv2.putText(frame, "ROI", (roi_x + 10, roi_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

        for oid, (centroid, area, brightness) in objects.items():
            cx, cy = centroid
            area_um2 = area_px * area_scale_factor
            label = f"ID{oid} A{area_um2:.0f}µm²"
            # to put the text including area and ID information on the video for each island:
            # cv2.putText(frame, label, (cx + 8, cy - 8),
                        # cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)  # smaller font + thinner
            # Small dot at centroid (less obtrusive)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 200), -1)

        out.write(frame)

    if frame_num % 100 == 0:
        print(f"Processed frame {frame_num}/{total_frames} — {len(objects)} islands tracked")

cap.release()
if WRITE_OUTPUT_VIDEO:
    out.release()

print("Analysis complete!")


# ====================== SAVE RESULTS TO CSV ======================
records = []
for oid, history in data.items():
    start_time = min(h["time_sec"] for h in history)
    for entry in history:
        records.append({
            "island_id": oid,
            "start_time_sec": round(start_time, 3),
            "time_sec": round(entry["time_sec"], 3),
            "area_px": entry["area_px"],
            "area_um2": entry["area_um2"],
            "mean_brightness": entry["mean_brightness"],
            "centroid_x": entry["centroid_x"],
            "centroid_y": entry["centroid_y"]
        })

df = pd.DataFrame(records)
df = df.sort_values(by=["island_id", "time_sec"]).reset_index(drop=True)
df.to_csv(OUTPUT_CSV, index=False)

print(f"Results saved → {OUTPUT_CSV}")
if WRITE_OUTPUT_VIDEO:
    print(f"Annotated video saved → {OUTPUT_VIDEO}")
print(f"Total islands detected: {len(data)}")

summary = df.groupby("island_id").agg(
    start_time=("start_time_sec", "first"),
    max_area_um2=("area_um2", "max"),
    final_area_um2=("area_um2", "last"),
    duration_sec=("time_sec", lambda x: x.max() - x.min())
).round(2)
print("Quick summary:")
print(summary)


    
# cap.release()
cv2.destroyAllWindows()