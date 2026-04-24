import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ====================== USER PARAMETERS ======================
CSV_PATH = "REACT004Visit5_March25_Straight1_Channel5_Exercise_NaHeparin_narrowTubing_height15cm_141ulMediumCBD_FIXED_islands_10frames_Mp4.csv"
# "REACT009Visit3_April2_Straight1_Channel6_Exercise_NaHeparin_narrowTubing_height15cm_141ulMediumCBD_islands_10frames.csv"
# "REACT004Visit5_March25_Straight1_Channel5_Exercise_NaHeparin_narrowTubing_height15cm_141ulMediumCBD_islands_10frames(in).csv"
# "REACT009Visit3_April2_Straight1_Channel5_REST_NaHeparin_narrowTubing_height15cm_141ulMediumCBD_islands_10frames.csv"
# "REACT004Visit5_March25_Straight1_Channel5_Exercise_NaHeparin_narrowTubing_height15cm_141ulMediumCBD_islands_10frames.csv"      # <--- CHANGE THIS to your v3 output file
base_name = Path(CSV_PATH).stem
OUTPUT_PLOT = base_name + "_total_thrombus_area_plot.png"
# "REACT004Visit5_March25_Straight1_Channel5_Exercise_NaHeparin_narrowTubing_height15cm_141ulMediumCBD_islands_10frames_total_thrombus_area_plot.png"
OUTPUT_CSV = base_name + "_total_area_over_time.csv"
# "REACT004Visit5_March25_Straight1_Channel5_Exercise_NaHeparin_narrowTubing_height15cm_141ulMediumCBD_islands_10frames_total_area_over_time.csv"

# Optional smoothing (set to 0 to disable)
SMOOTH_WINDOW = 1000                        # moving average window in frames (e.g. 5 = 5-frame smoothing)
# ============================================================

print("Loading CSV from v3 tracker...")
df = pd.read_csv(CSV_PATH)

# Ensure correct data types
df["time_sec"] = pd.to_numeric(df["time_sec"], errors="coerce")
df["area_px"] = pd.to_numeric(df["area_px"], errors="coerce")
df["area_um2"] = pd.to_numeric(df["area_um2"], errors="coerce")


# Drop any corrupted rows
df = df.dropna(subset=["time_sec", "area_px", "area_um2"])

print(f"   Loaded {len(df):,} rows | {df['island_id'].nunique()} unique islands")
print(f"   Time range: {df['time_sec'].min():.2f} s → {df['time_sec'].max():.2f} s")

# ====================== COMPUTE TOTAL AREA PER FRAME ======================
print("Computing total thrombus area per time point...")

# Group by time_sec and sum areas of all islands present at that time
total_area_df = (
    df.groupby("time_sec", as_index=False)["area_um2"]
    .sum()
    .rename(columns={"area_um2": "total_area_um2"})
)

# Optional: add number of active islands per frame
active_islands = df.groupby("time_sec")["island_id"].nunique().reset_index()
total_area_df = total_area_df.merge(active_islands, on="time_sec", how="left")
total_area_df = total_area_df.rename(columns={"island_id": "num_active_islands"})

# Sort by time (just in case)
total_area_df = total_area_df.sort_values("time_sec").reset_index(drop=True)

# Optional smoothing
if SMOOTH_WINDOW > 1:
    print(f"   Applying {SMOOTH_WINDOW}-frame moving average smoothing...")
    total_area_df["total_area_um2_smooth"] = (
        total_area_df["total_area_um2"]
        .rolling(window=SMOOTH_WINDOW, center=True, min_periods=1)
        .mean()
    )
else:
    total_area_df["total_area_um2_smooth"] = total_area_df["total_area_um2"]

# ====================== SAVE SUMMARY CSV ======================
total_area_df.to_csv(OUTPUT_CSV, index=False)
print(f"Summary saved → {OUTPUT_CSV}")

# ====================== PLOT ======================
plt.figure(figsize=(12, 7))

# Raw data (light)
plt.plot(
    total_area_df["time_sec"], 
    total_area_df["total_area_um2"], 
    color="#00d4ff", 
    alpha=0.25, 
    linewidth=1.2,
    label="Raw total area"
)

# Smoothed curve (main line)
plt.plot(
    total_area_df["time_sec"], 
    total_area_df["total_area_um2_smooth"], 
    color="#00d4ff", 
    linewidth=3.5,
    label=f"Total thrombus area (smoothed, window={SMOOTH_WINDOW})"
)

# plt.title("Total Thrombus Area Over Time\n(all islands combined)", fontsize=18, pad=20)
plt.xlabel("Time (seconds)", fontsize=24)
plt.ylabel("Total Area (um2)", fontsize=24)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=20)

# Nice formatting
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# Optional: highlight final value
final_time = total_area_df["time_sec"].iloc[-1]
final_area = total_area_df["total_area_um2_smooth"].iloc[-1]
plt.annotate(
    f"Final area: {final_area:,.0f} um2",
    xy=(final_time, final_area),
    xytext=(final_time*0.8, final_area*1.1),
    arrowprops=dict(arrowstyle="->", color="#ff8800"),
    fontsize=18,
    color="#ff8800"
)

plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches="tight")
plt.show()

print(f"Plot saved → {OUTPUT_PLOT}")

# Quick stats
print("\n Quick statistics:")
print(total_area_df[["total_area_um2", "total_area_um2_smooth", "num_active_islands"]].describe().round(2))
print(f"\nMaximum total area: {total_area_df['total_area_um2'].max():,.0f} um2")