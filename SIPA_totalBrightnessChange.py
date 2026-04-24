import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ====================== PARAMETERS ======================
VIDEO_PATH = "C:/Users/rojan/Documents/FSU/codes/PythonCodes/SIPAVideos/REACT009Visit3_April2_Straight1_Channel5_REST_NaHeparin_narrowTubing_height15cm_141ulMediumCBD.avi"
# "C:/Users/rojan/Documents/FSU/codes/PythonCodes/SIPAVideos/REACT009Visit3_April2_Straight1_Channel6_Exercise_NaHeparin_narrowTubing_height15cm_141ulMediumCBD.avi"
# "C:/Users/rojan/Documents/FSU/codes/PythonCodes/SIPAVideos/REACT004Visit5_March25_Straight1_Channel5_Exercise_NaHeparin_narrowTubing_height15cm_141ulMediumCBD.avi"
OUTPUT_CSV = "REACT009Visit3_April2_Straight1_Channel5_REST_NaHeparin_narrowTubing_height15cm_141ulMediumCBD_roi_brightness.csv"
OUTPUT_PLOT1 = "REACT009Visit3_April2_Straight1_Channel5_REST_NaHeparin_narrowTubing_height15cm_141ulMediumCBD_MeanBrightness.png"
OUTPUT_PLOT2 = "REACT009Visit3_April2_Straight1_Channel5_REST_NaHeparin_narrowTubing_height15cm_141ulMediumCBD_ExcessBrightness.png"

# Background model
NUM_BACKGROUND_FRAMES = 500             # must be BEFORE any islands appear

# ROI (interactive)
USE_INTERACTIVE_ROI = True

# NEW: Robust reading controls
REOPEN_EVERY_N_FRAMES = 8000           # re-open capture periodically (prevents index crash)
MAX_RECOVERY_RETRIES = 100
# =========================================================

# ====================== ROBUST VIDEO READER ======================
def create_cap():
    """Create VideoCapture with FFMPEG backend (best for long AVIs)"""
    return cv2.VideoCapture(VIDEO_PATH, cv2.CAP_FFMPEG)
cap = create_cap()
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# Set position to the very end (after the last frame)
cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames)
# Get the timestamp of the last frame in milliseconds
duration_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
duration_sec = duration_ms / 1000.0

fps = total_frames/duration_sec +1 #cap.get(cv2.CAP_PROP_FPS) or 30.0


frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video loaded — {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames @ {fps:.2f} fps")

# ====================== ROI SELECTION ======================
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
if USE_INTERACTIVE_ROI:
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read first frame")
    print("Draw box around the stenotic channel (press ENTER)")
    roi = cv2.selectROI("Select ROI", first_frame, False)
    cv2.destroyAllWindows()
    roi_x, roi_y, roi_w, roi_h = map(int, roi)
    if roi_w == 0 or roi_h == 0:
        roi_x = roi_y = 0
        roi_w, roi_h = frame_width, frame_height
else:
    roi_x, roi_y, roi_w, roi_h = 100, 200, 600, 300  # hard-code if you prefer

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# ====================== BUILD BACKGROUND ======================
print(f"Building background from first {NUM_BACKGROUND_FRAMES} frames...")
bg_frames = []
cap = create_cap()
for i in range(NUM_BACKGROUND_FRAMES):
    ret, frame = cap.read()
    if not ret:
        break
    roi_frame = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY) if len(roi_frame.shape) == 3 else roi_frame.copy()
    bg_frames.append(gray.astype(np.float32))

bg = np.median(bg_frames, axis=0)   # robust background

cap.set(cv2.CAP_PROP_POS_FRAMES, NUM_BACKGROUND_FRAMES)   # reset video
print("Background ready")

# ====================== COMPUTE BRIGHTNESS METRICS ======================
print("Analyzing brightness frame-by-frame...")

cap = create_cap()
times = []
mean_brightness = []
excess_brightness = []   # the key metric for thrombus growth

recovery_attempts = 0
last_successful_frame = 0

frame_num = NUM_BACKGROUND_FRAMES
while cap.isOpened(): #True:
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
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)   # jump to the next expected frame
        recovery_attempts += 1

        # Try one more read immediately
        ret, frame = cap.read()
        if not ret:
            print("Still failing after reopen — stopping.")
            break

        

    frame_num += 1
    last_successful_frame = frame_num

    # Periodic reopen (prevents the 2300-frame crash)
    if frame_num % REOPEN_EVERY_N_FRAMES == 0:
        print(f"Periodic reopen at frame {frame_num} (to keep index stable)")
        cap.release()
        cap = create_cap()
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        continue   # skip processing this frame (we just reopened)
        
    time_sec = frame_num / fps

    # Crop ROI
    roi_frame = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY) if len(roi_frame.shape) == 3 else roi_frame.copy()
    gray_float = gray.astype(np.float32)

    # Mean brightness (average intensity)
    mean_b = np.mean(gray)

    # Excess brightness (only the brighter parts caused by islands)
    diff = cv2.absdiff(gray_float, bg)          # absolute difference
    excess = np.sum(np.where(gray_float > bg, diff, 0))   # only positive changes

    times.append(time_sec)
    mean_brightness.append(mean_b)
    excess_brightness.append(excess)

    if frame_num % 100 == 0:
        print(f"Processed frame {frame_num}")

cap.release()

# ====================== SAVE CSV ======================
df = pd.DataFrame({
    "time_sec": times,
    "mean_brightness": mean_brightness,
    "excess_brightness": excess_brightness,
    "excess_brightness_normalized": np.array(excess_brightness) / (roi_w * roi_h)
})

df.to_csv(OUTPUT_CSV, index=False)
print(f"Data saved → {OUTPUT_CSV}")

# ====================== PLOT ======================

fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(times, mean_brightness, color='#00d4ff', label='Mean Brightness (overall)', linewidth=2)
ax1.set_xlabel('Time (seconds)', fontsize=14)
ax1.set_ylabel('Mean Brightness (0–255)', color='#00d4ff', fontsize=14)
ax1.tick_params(axis='y', labelcolor='#00d4ff')
ax1.axvline(77.4, color='r', linestyle='--')
ax1.axvline(156, color='b', linestyle='--')
ax1.axvline(235, color='b', linestyle='--')
ax1.axvline(319, color='b', linestyle='--')
ax1.axvline(395, color='b', linestyle='--')
plt.title('ROI Brightness Over Time — Thrombotic Island Growth', fontsize=16, pad=20)
fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_PLOT1, dpi=300)
plt.show()

fig, ax2 = plt.subplots(figsize=(12, 6))
# ax2 = ax1.twinx()
ax2.plot(times, excess_brightness, color='#ff8800', label='Excess Brightness (islands only)', linewidth=2.5)
ax2.set_ylabel('Excess Brightness (sum of brighter pixels)', color='#ff8800', fontsize=14)
ax2.tick_params(axis='y', labelcolor='#ff8800')
ax2.axvline(77.4, color='r', linestyle='--')
ax2.axvline(300, color='b', linestyle='--')
ax2.axvline(235, color='b', linestyle='--')
plt.title('ROI Brightness Over Time — Thrombotic Island Growth', fontsize=16, pad=20)
fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_PLOT2, dpi=300)
plt.show()

print("\nQuick summary:")
print(df.describe().round(2))