import cv2
import os
from deepface import DeepFace
import glob
import time

# Define the local directory
base_dir = os.path.expanduser("~/Desktop/vision")

# Specify your video path and reference images
video_path = os.path.join(base_dir, "a1.mp4")  # Using a1.mp4; change to a2.mp4 if needed
staff_images_all = glob.glob(os.path.join(base_dir, "person*.jpg"))

# Define the structure for the four persons with check status
persons = {
    "person1": {"check": True, "file": os.path.join(base_dir, "person1.jpg")},
    "person2": {"check": True, "file": os.path.join(base_dir, "person2.jpg")},
    "person3": {"check": True, "file": os.path.join(base_dir, "person3.jpg")},
    "person4": {"check": True, "file": os.path.join(base_dir, "person4.jpg")}
}

# Filter staff images based on check status
staff_images = [p["file"] for p in persons.values() if p["check"] and os.path.exists(p["file"])]

# Check if video file exists
print("Checking video path:", os.path.exists(video_path))
if not os.path.exists(video_path):
    print("Error: Video file not found. Please verify the path.")
    print("Directory contents:", os.listdir(base_dir))
    raise FileNotFoundError("Video path invalid")

# Check if reference images exist
print("Checking persons to process:", [name for name, info in persons.items() if info["check"]])
print("Found reference images:", staff_images)
if not staff_images:
    print("Error: No reference images selected or found.")
    raise FileNotFoundError("No staff images found")

# Open the video and verify it works
vidcap = cv2.VideoCapture(video_path)
if not vidcap.isOpened():
    print("Error: Could not open video file.")
    raise ValueError("Video file cannot be opened")
else:
    print("Video opened successfully.")
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    print(f"Total frames in video: {frame_count}")
    print(f"Video FPS: {fps}")

# Dictionary to store detection results
detected_staff = {}

# Process first 10 seconds at 5 FPS
max_seconds = 10
target_fps = 1  # 5 FPS
max_frames = int(fps * max_seconds)
frame_interval = max(1, int(fps / target_fps))
frames_to_process = range(0, max_frames, frame_interval)
total_frames = len(frames_to_process)

print(f"Processing first {max_seconds} seconds at {target_fps} FPS ({total_frames} frames)...")
start_time = time.time()

for i, frame_num in enumerate(frames_to_process):
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    success, image = vidcap.read()
    if not success:
        print(f"Failed to read frame {frame_num}")
        break

    for staff_img in staff_images:
        try:
            result = DeepFace.verify(img1_path=image, img2_path=staff_img, 
                                    model_name="Facenet", 
                                    distance_metric="cosine", 
                                    enforce_detection=False)
            if result["verified"]:
                staff_name = os.path.basename(staff_img).split(".")[0]
                detected_staff[frame_num] = staff_name
                print(f"Match found: {staff_name} at frame {frame_num} (time: {frame_num/fps:.1f}s)")
                break
        except Exception as e:
            print(f"Error processing frame {frame_num} with {staff_img}: {e}")

    # Calculate and display progress with elapsed time
    percent_complete = (i + 1) / total_frames * 100
    elapsed_time = time.time() - start_time
    print(f"Progress: {percent_complete:.1f}% ({i + 1}/{total_frames} frames) | Elapsed time: {elapsed_time:.1f}s", end="\r")
    if detected_staff:
        print(f"\nCurrent detections: {list(set(detected_staff.values()))}")

# Final newline to avoid overwriting last progress line
print()

# Release the video capture object
vidcap.release()
end_time = time.time()
total_time = end_time - start_time

# Summarize detections for the four persons
detection_summary = {name: name in detected_staff.values() for name in persons.keys()}

print(f"Processing complete. Total frames processed: {total_frames}")
print(f"Final time taken: {total_time:.1f} seconds")
print("Detection summary:")
for name, detected in detection_summary.items():
    status = "Detected" if detected else "Not detected"
    print(f"  {name}: {status}")
print("Detected staff in video:", list(set(detected_staff.values())))