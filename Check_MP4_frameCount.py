import cv2
cap = cv2.VideoCapture("C:/Users/rojan/Documents/FSU/codes/PythonCodes/SIPAVideos/REACT009Visit3_April2_Straight1_Channel6_Exercise_NaHeparin_narrowTubing_height15cm_141ulMediumCBD_FIXED.mp4", cv2.CAP_FFMPEG)
    # "C:/Users/rojan/Documents/FSU/codes/PythonCodes/SIPAVideos/REACT004Visit5_March25_Straight1_Channel5_Exercise_NaHeparin_narrowTubing_height15cm_141ulMediumCBD_FIXED.mp4", cv2.CAP_FFMPEG)
# "C:/Users/rojan/Documents/FSU/codes/PythonCodes/SIPAVideos/REACT009Visit3_April2_Straight1_Channel6_Exercise_NaHeparin_narrowTubing_height15cm_141ulMediumCBD_FIXED2.mp4"
count = 0
while True:
    ret, _ = cap.read()
    if not ret: break
    count += 1
    if count % 10000 == 0: print(count)
print("Total readable frames:", count)
cap.release()