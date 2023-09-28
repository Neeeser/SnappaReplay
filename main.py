import os

import cv2
import time
from collections import deque

# Path where videos are storeds
root_video_path = 'videos/'
timestamp = time.strftime("%Y%m%d-%H%M%S")
filename = f'Game_{timestamp}/'
video_path = root_video_path + filename
full_video_name = video_path + 'full_game.avi'

# Check if those directory exists, if not create them
if not os.path.exists(root_video_path):
    os.makedirs(root_video_path)
if not os.path.exists(video_path):
    os.makedirs(video_path)


# Initialize the camera
cap = cv2.VideoCapture(0)  # 0 for the default camera, change if you have multiple cameras

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Adjust frame rate and duration as needed
frame_rate = 20.0  # frames per second
duration = 30  # duration to save in seconds


# Define a deque to hold the last 'duration' seconds of frames
buffer = deque(maxlen=int(frame_rate) * duration)

# Define the video writer for the full game
out_full = cv2.VideoWriter(full_video_name, fourcc, frame_rate, (frame_width, frame_height))


def detect_dice(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use thresholding or Canny edge detection
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by size to detect dice
    for contour in contours:
        if cv2.contourArea(contour) < 5000:  # adjust this value as per your requirement
            continue

        # Draw a rectangle around detected dice
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame

def draw_scoreboard(frame, score_left, score_right, elapsed_time):
    h, w, _ = frame.shape

    # Define properties of the scoreboard
    height = 100  # Height of the scoreboard
    color = (42,50,155) # White color for the scoreboard
    thickness = -1  # Fill the rectangle

    # Draw the scoreboard rectangle
    scoreboard = cv2.rectangle(frame, (0, h - height), (w, h), color, thickness)

    # Define font and scale
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_color = (58,224,168)#(224, 168,58)#(168, 58, 224)  # Black color for the text
    font_thickness = 2

    # Position the scores on the scoreboard
    left_score_position = (int(w * 0.25), h - int(height * 0.5))
    right_score_position = (int(w * 0.75), h - int(height * 0.5))
    time_position = (int(w * 0.5), h - int(height * 0.5))

    # Put the scores and elapsed time on the scoreboard
    cv2.putText(frame, str(score_left), left_score_position, font, font_scale, font_color, font_thickness, cv2.LINE_AA)
    cv2.putText(frame, str(score_right), right_score_position, font, font_scale, font_color, font_thickness,
                cv2.LINE_AA)
    cv2.putText(frame, elapsed_time, time_position, font, font_scale, font_color, font_thickness, cv2.LINE_AA)

    return frame

def put_text(frame):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "American Snappa League"
    color = (0, 0, 255)  # Red
    thickness = 2
    x = int(frame_width/2) - 200  # Adjust position as needed
    y = int(frame_height/2)  # Adjust position as needed


    # Creating bold effect by overlaying the text multiple times with slight offsets
    cv2.putText(frame, text, (x, y), font, 1, color, thickness + 1)
    cv2.putText(frame, text, (x - 1, y), font, 1, color, thickness)
    cv2.putText(frame, text, (x + 1, y), font, 1, color, thickness)
    cv2.putText(frame, text, (x, y - 1), font, 1, color, thickness)
    cv2.putText(frame, text, (x, y + 1), font, 1, color, thickness)
    return frame


print("Recording... Press 'q' to stop, 's' to save the last 30 seconds.")
start_time = time.time()
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Draw the scoreboard
        elapsed_time_sec = int(time.time() - start_time)
        elapsed_time = f"{elapsed_time_sec // 60:02d}:{elapsed_time_sec % 60:02d}"  # Convert to MM:SS format
        score_left = 2  # Replace with the actual score
        score_right = 5  # Replace with the actual score

        frame = detect_dice(frame)
        frame = draw_scoreboard(frame, score_left, score_right, elapsed_time)

        # # Save to full game video
        # out_full.write(frame)

        # Save to buffer
        buffer.append(frame)

        # Show the frame (optional)
        cv2.imshow('Frame', frame)

        key = cv2.waitKey(1) & 0xFF

        # Check for quit command
        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = video_path + f'saved_clip_{timestamp}.avi'
            out_clip = cv2.VideoWriter(filename, fourcc, frame_rate, (frame_width, frame_height))
            for index, frame in enumerate(buffer):
                if index < frame_rate * 5:  # for the first 3 seconds
                    frame = put_text(frame)
                out_clip.write(frame)
            out_clip.release()
            print(f"Saved the last {duration} seconds to {filename}")

            # Replay the saved clip
            cap_clip = cv2.VideoCapture(filename)
            while True:
                ret, frame = cap_clip.read()
                if not ret:
                    break
                cv2.imshow('Frame', frame)
                if cv2.waitKey(int(1000 / frame_rate)) & 0xFF == ord('q'):
                    break
            cap_clip.release()

            # Clear the buffer
            buffer.clear()

        # If not in replay mode, write the frame to full_game.avi
        else:
            out_full.write(frame)

finally:
    # Release resources
    cap.release()
    out_full.release()
    cv2.destroyAllWindows()

print("Recording stopped.")
