import os
import cv2
import face_recognition
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Path to the MKV video file
video_path = "video.mkv"

# Load reference image of the person
ref_image = face_recognition.load_image_file("image.png")
ref_encoding = face_recognition.face_encodings(ref_image)[0]

# Create a directory to store the extracted frames
output_dir = "frames"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Open the video file using OpenCV
cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Define a function to process a single frame and return the result
def process_frame(frame):
    # Find all the faces in the frame and get their encodings
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Compare the face encodings with the reference encoding
    for j, face_encoding in enumerate(face_encodings):
        result = face_recognition.compare_faces([ref_encoding], face_encoding)

        # If the faces match, save the frame as a PNG image
        if result[0] == True:
            frame_path = os.path.join(output_dir, f"frame_{i}.png")
            cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# Set up multiprocessing
num_processes = min(cpu_count(), frame_count)
pool = Pool(num_processes)

# Set up the progress bar
pbar = tqdm(total=frame_count, desc="Processing video", unit="frame")

# Iterate through all the frames in the video
for i in range(frame_count):
    ret, frame = cap.read()
    if not ret:
        break

    pool.apply_async(process_frame, (frame,))

    # Update the progress bar
    pbar.update(1)

# Close the progress bar and release the video file
pbar.close()
cap.release()

# Wait for all the frame processing to complete
pool.close()
pool.join()

print("Extraction complete.")
