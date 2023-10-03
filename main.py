import os
import numpy as np
import cv2
import time
from collections import deque
from multiprocessing import Process



# Path where videos are stored
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
print(f"Frame width: {frame_width} | Frame height: {frame_height}")

# Adjust frame rate and duration as needed
frame_rate = 20.0  # frames per second
duration = 30  # duration to save in seconds
frame_display_time = int(1000 / frame_rate)

# Define a deque to hold the last 'duration' seconds of frames
buffer = deque(maxlen=int(frame_rate) * duration)

# Define the video writer for the full game
out_full = cv2.VideoWriter(full_video_name, fourcc, frame_rate, (frame_width, frame_height))

def save_video(buffer, filename):
    out_clip = cv2.VideoWriter(filename, fourcc, frame_rate, (frame_width, frame_height))
    for frame in buffer:
        out_clip.write(frame)
    out_clip.release()
    print(f"Saved to {filename}")

def overlay_transition(frame, gradient_img, alpha):
    return cv2.addWeighted(gradient_img, alpha, frame, 1 - alpha, 0)


def create_gradient_image(width, height):
    # Define the start and end colors (red and gold in BGR format)
    start_color = [0, 0, 255]  # red
    end_color = [0, 215, 255]  # gold

    # Create a gradient image
    gradient_img = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(width):
        alpha = i / width
        color = [int((1 - alpha) * start + alpha * end) for start, end in zip(start_color, end_color)]
        gradient_img[:, i] = color

    return gradient_img

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

    # Flips the frame horizontally
    frame = cv2.flip(frame, 1)

    # Define properties of the scoreboard
    height = 80  # Height of the scoreboard
    color = (42, 50, 155)  # Color for the scoreboard
    alpha = 0.7  # Transparency factor [0, 1] where 0 is completely transparent and 1 is completely opaque

    # Create a copy of the original frame
    overlay = frame.copy()

    # Draw the transparent rectangle on the copy of the original frame
    cv2.rectangle(overlay, (0, h - height), (w, h), color, -1)

    # Blend the original frame with the overlay using alpha
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Define font and scale
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_color = (58, 224, 168)  # Color for the text
    font_thickness = 2

    # Position the scores on the scoreboard
    left_score_position = (int(w * 0.15 - font_scale * 10), h - int(height * 0.5 - font_scale * 10))
    right_score_position = (int(w * 0.85 + font_scale * 10), h - int(height * 0.5 - font_scale * 10))
    time_position = (int(w * 0.5 - font_scale * 10), h - int(height * 0.5 - font_scale * 10))

    # Put the scores and elapsed time on the scoreboard
    cv2.putText(frame, str(score_left), left_score_position, font, font_scale, font_color, font_thickness, cv2.LINE_AA)
    cv2.putText(frame, str(score_right), right_score_position, font, font_scale, font_color, font_thickness,
                cv2.LINE_AA)
    cv2.putText(frame, elapsed_time, time_position, font, font_scale, font_color, font_thickness, cv2.LINE_AA)

    return frame


def put_text(frame):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "Replay"
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


def overlay_image(frame, overlay, position=(0, 0), scale=1):
    # Resize overlay image
    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)

    # Get the region of interest (ROI) in the frame
    h, w, _ = overlay.shape
    x, y = position
    roi = frame[y:y + h, x:x + w]

    # Extract the alpha channel from the overlay image and create an inverse alpha mask
    alpha = overlay[:, :, 3] / 255.0
    inverse_alpha = 1.0 - alpha

    # Blend the overlay image with the ROI based on the alpha channel
    for c in range(0, 3):
        roi[:, :, c] = (alpha * overlay[:, :, c] + inverse_alpha * roi[:, :, c])

    return frame


def overlay_replay_banner(frame, overlay_width_percentage=0.3):
    # Define properties of the replay banner
    overlay_path = "imgs/ReplayOverlay.png"

    # Desired width of the overlay as a percentage of frame's width
    desired_width = int(frame_width * overlay_width_percentage)

    # Load the overlay image
    overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
    original_height, original_width, _ = overlay.shape

    # Calculate the scale factor
    scale = desired_width / original_width

    # Resize the overlay image
    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    overlay_height, overlay_width, _ = overlay.shape


    # Calculate the position to center the overlay on the frame
    x = int((frame_width - overlay_width) / 2)  # Used float division here and then converted to int
    #y = int((frame_height - overlay_height) / 10)  # Adjust the denominator as needed
    y = 0

    position = (x, y)

    # Overlay the replay banner on the frame
    frame = overlay_image(frame, overlay, position)

    return frame


def main():
    print("Recording... Press 'q' to stop, 's' to save the last 30 seconds.")
    start_time = time.time()
    score_left = 0  # Replace with the actual score
    score_right = 0  # Replace with the actual score
    gradient_img = create_gradient_image(frame_width, frame_height)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Draw the scoreboard
            elapsed_time_sec = int(time.time() - start_time)
            elapsed_time = f"{elapsed_time_sec // 60:02d}:{elapsed_time_sec % 60:02d}"  # Convert to MM:SS format


            #frame = detect_dice(frame)
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
                p = Process(target=save_video, args=(list(buffer), filename,))
                p.start()

                # Define the number of frames for the transition
                num_transition_frames = 60  # Adjust as needed

                # Replay the frames in the buffer with the transition
                for index, frame in enumerate(buffer):
                    if index < num_transition_frames:  # Transition period
                        alpha = 1 - (index / num_transition_frames)  # This line is modified to go from 1 to 0
                        frame = overlay_transition(frame, gradient_img, alpha)

                    if index >= num_transition_frames / 2 :  # for the first 3 seconds post-transition
                        frame = overlay_replay_banner(frame)

                    cv2.imshow('Frame', frame)
                    if cv2.waitKey(int(1000 / frame_rate)) & 0xFF == ord('q'):
                        break

                p.join()
                buffer.clear()


            elif key == ord('1'):
                score_left += 1
            elif key == ord('2'):
                score_right += 1

            # If not in replay mode, write the frame to full_game.avi
            else:
                out_full.write(frame)

    # Make it except for keyboard interrupt
    except KeyboardInterrupt:
        pass
    finally:
        # Release resources
        cap.release()
        out_full.release()
        cv2.destroyAllWindows()

    print("Recording stopped.")


if __name__ == "__main__":
    main()