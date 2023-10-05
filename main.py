import os
import numpy as np
import cv2
import time
from collections import deque
from multiprocessing import Process
from tools import save_video

## Define keyboard Macros
TEAM1_PLAYER1_HIT = 'w'
TEAM1_PLAYER1_SINK = 'e'
TEAM1_PLAYER1_MISS = 'r'
TEAM1_PLAYER1_DROP = 't'

TEAM1_PLAYER2_HIT = 'z'
TEAM1_PLAYER2_SINK = 'x'
TEAM1_PLAYER2_MISS = 'c'
TEAM1_PLAYER2_DROP = 'v'

TEAM2_PLAYER1_HIT = 'u'
TEAM2_PLAYER1_SINK = 'i'
TEAM2_PLAYER1_MISS = 'o'
TEAM2_PLAYER1_DROP = 'p'

TEAM2_PLAYER2_HIT = 'n'
TEAM2_PLAYER2_SINK = 'm'
TEAM2_PLAYER2_MISS = ','
TEAM2_PLAYER2_DROP = '.'

BACKSPACE_KEY = ord('\x08')  # or simply 8
BACKSPACE_KEY_MAC = 127



# Define the key mappings
ACTIONS = {
    ord('w'): {'team': 'Team1', 'player': 'PlayerOne', 'action': 'Hits'},
    ord('e'): {'team': 'Team1', 'player': 'PlayerOne', 'action': 'Sinks'},
    ord('r'): {'team': 'Team1', 'player': 'PlayerOne', 'action': 'Misses'},
    ord('t'): {'team': 'Team1', 'player': 'PlayerOne', 'action': 'Drops'},

    ord('z'): {'team': 'Team1', 'player': 'PlayerTwo', 'action': 'Hits'},
    ord('x'): {'team': 'Team1', 'player': 'PlayerTwo', 'action': 'Sinks'},
    ord('c'): {'team': 'Team1', 'player': 'PlayerTwo', 'action': 'Misses'},
    ord('v'): {'team': 'Team1', 'player': 'PlayerTwo', 'action': 'Drops'},

    ord('u'): {'team': 'Team2', 'player': 'PlayerOne', 'action': 'Hits'},
    ord('i'): {'team': 'Team2', 'player': 'PlayerOne', 'action': 'Sinks'},
    ord('o'): {'team': 'Team2', 'player': 'PlayerOne', 'action': 'Misses'},
    ord('p'): {'team': 'Team2', 'player': 'PlayerOne', 'action': 'Drops'},

    ord('n'): {'team': 'Team2', 'player': 'PlayerTwo', 'action': 'Hits'},
    ord('m'): {'team': 'Team2', 'player': 'PlayerTwo', 'action': 'Sinks'},
    ord(','): {'team': 'Team2', 'player': 'PlayerTwo', 'action': 'Misses'},
    ord('.'): {'team': 'Team2', 'player': 'PlayerTwo', 'action': 'Drops'}
}

def update_score(team_info):
    # loop through the teams and update team points based on these rules one 1 is +1 points 1 sink is +1 points
    for team in team_info:
        team_info[team]['TeamPoints'] = team_info[team]['PlayerOne']['Hits'] + team_info[team]['PlayerOne']['Sinks'] + team_info[team]['PlayerTwo']['Hits'] + team_info[team]['PlayerTwo']['Sinks']
        team_info[team]['PlayerOne']['Stats'] = team_info[team]['PlayerOne']['Hits'] + (team_info[team]['PlayerOne']['Sinks']*3) - (team_info[team]['PlayerOne']['Misses']/2) - team_info[team]['PlayerOne']['Drops']
        team_info[team]['PlayerTwo']['Stats'] = team_info[team]['PlayerTwo']['Hits'] + (team_info[team]['PlayerTwo']['Sinks']*3) - (team_info[team]['PlayerTwo']['Misses']/2) - team_info[team]['PlayerTwo']['Drops']


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
# cap = cv2.VideoCapture(0)  # 0 for the default camera, change if you have multiple cameras
while True:
    try:
        #stream_url = "rtsp://admin:4647@andrew.local:8554/live"
        stream_url = "http://admin:4647@andrew.local:8081/video"
        cap = cv2.VideoCapture(stream_url)
        if cap.isOpened():
            break
    except (cv2.error, ValueError) as e:
        print("Error: Could not open camera.")
        time.sleep(1)


#desired_frame_rate = 60  # frames per second


#cap.set(cv2.CAP_PROP_FPS, desired_frame_rate)  # Set an extremely high value


# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

cv2.namedWindow('Frame', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)




if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'XVID')

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
print(f"Frame width: {frame_width} | Frame height: {frame_height}")

# Adjust frame rate and duration as needed
duration = 15  # duration to save in seconds

actual_fps = cap.get(cv2.CAP_PROP_FPS)
frame_rate = actual_fps
frame_display_time = int(1000 / frame_rate)


#cap.set(cv2.CAP_PROP_FPS, frame_rate)

# Define a deque to hold the last 'duration' seconds of frames
buffer = deque(maxlen=int(frame_rate) * duration)

# Define the video writer for the full game
out_full = cv2.VideoWriter(full_video_name, fourcc, frame_rate, (frame_width, frame_height))

# def save_video(buffer, filename):
#     print(len(buffer)/frame_rate)
#     out_clip = cv2.VideoWriter(filename, fourcc, frame_rate, (frame_width, frame_height))
#     for frame in buffer:
#         out_clip.write(frame)
#
#
#     out_clip.release()
#     print(f"Saved to {filename}")

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


def draw_scoreboard(frame, team_info, elapsed_time):
    h, w, _ = frame.shape

    # Flips the frame horizontally
    frame = cv2.flip(frame, 1)

    # Define properties of the scoreboard
    height = frame_height // 8
    color = (42, 50, 155)
    alpha = 0.7

    # Create an overlay
    overlay = frame.copy()

    # Draw a rectangle for the scoreboard
    cv2.rectangle(overlay, (0, h - height), (w, h), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Calculate font scale based on rectangle height
    num_lines = 3
    max_line_height = height / (num_lines + 1)  # +1 for some spacing
    font_scale = max_line_height / 25  # Adjust the denominator (25 in this case) based on how the text looks

    # Add more space between lines
    spacing_factor = 1.2
    max_line_height *= spacing_factor  # Increase space between lines


    # Define font and colors
    font = cv2.FONT_HERSHEY_COMPLEX
    header_color = (255, 255, 255)
    text_color = (58, 224, 168)
    font_thickness = int(frame_width * 0.001)

    # Calculate positions
    left_position = int(w * 0.01)
    right_position = int(w * 0.55)
    vertical_position = h - height + int(max_line_height)  # Starting at the first line


    # Draw Team's info for both teams
    for team, data in team_info.items():
        # Team's Name and Points
        team_text = f"{data['TeamName']} - {data['TeamPoints']} Points"

        if team == 'Team1':
            position = left_position
        else:
            text_width, _ = cv2.getTextSize(team_text, font, font_scale, font_thickness)[0]
            position = w - int(w * 0.01) - text_width

        cv2.putText(frame, team_text, (position, vertical_position), font, font_scale, header_color, font_thickness,
                    cv2.LINE_AA)
        vertical_position += int(max_line_height)

        # Draw Players' stats
        for player in ['PlayerOne', 'PlayerTwo']:
            player_info = data[player]
            formatted_stats = "{:+.2f}".format(player_info["Stats"])
            player_text = f"{player_info['PlayerName']} | {formatted_stats} H: {player_info['Hits']} M: {player_info['Misses']} S: {player_info['Sinks']} D: {player_info['Drops']}"

            if team == 'Team1':
                position = left_position
                player_text = f"H: {player_info['Hits']} M: {player_info['Misses']} S: {player_info['Sinks']} D: {player_info['Drops']} {formatted_stats} | {player_info['PlayerName']}"

            else:
                text_width, _ = cv2.getTextSize(player_text, font, font_scale, font_thickness)[0]
                position = w - int(w * 0.01) - text_width

            cv2.putText(frame, player_text, (position, vertical_position), font, font_scale, text_color, font_thickness,
                        cv2.LINE_AA)
            vertical_position += int(max_line_height)

        # Reset vertical position for the second team
        vertical_position = h - height + int(max_line_height)

    # ...

    # Display elapsed time at the bottom center
    time_position = (w // 2 - int(frame_width * 0.05), h - int(frame_height * 0.01))
    cv2.putText(frame, elapsed_time, time_position, font, font_scale, header_color, font_thickness, cv2.LINE_AA)

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

def undo_last_action(action_log):
    if action_log:
        last_action = action_log.pop()  # Get the last action
        team, player, action = last_action
        team_info[team][player][action] -= 1  # Decrement the respective action for the player


team_info = {
                'Team1': {
                    'TeamName': "Team A",
                    'PlayerOne': {'PlayerName': "Joe", 'Hits': 0, 'Misses': 0, 'Sinks': 0, 'Drops': 0, 'Stats': 0.0},
                    'PlayerTwo': {'PlayerName': "Mama", 'Hits': 0, 'Misses': 0, 'Sinks': 0, 'Drops': 0, 'Stats': 0.0},
                    'TeamPoints': 0
                },
                'Team2': {
                    'TeamName': "Team B",
                    'PlayerOne': {'PlayerName': "John", 'Hits': 0, 'Misses': 0, 'Sinks': 0, 'Drops': 0, 'Stats': 0.0},
                    'PlayerTwo': {'PlayerName': "Doe", 'Hits': 0, 'Misses': 0, 'Sinks': 0, 'Drops': 0, 'Stats': 0.0},
                    'TeamPoints': 0
                }
            }

def main():
    print("Recording... Press 'q' to stop, 's' to save the last 30 seconds.")
    start_time = time.time()
    score_left = 0  # Replace with the actual score
    score_right = 0  # Replace with the actual score
    gradient_img = create_gradient_image(frame_width, frame_height)
    # Log of all actions for undo functionality
    action_log = []
    try:
        while True:
            if not cap.isOpened():
                print("Capture not opened")
                cap.open(stream_url)
            ret, frame = cap.read()

            if not ret:
                print("Failed to grab frame")
                break

            # Draw the scoreboard
            elapsed_time_sec = int(time.time() - start_time)
            elapsed_time = f"{elapsed_time_sec // 60:02d}:{elapsed_time_sec % 60:02d}"  # Convert to MM:SS format



            update_score(team_info)
            frame = draw_scoreboard(frame, team_info, elapsed_time)
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
                # p = Process(target=save_video, args=(list(buffer), filename,frame_rate, frame_width, frame_height,))
                # p.start()
                #out_clip = cv2.VideoWriter(filename, fourcc, frame_rate, (frame_width, frame_height))

                # Define the number of frames for the transition
                num_transition_frames = 60  # Adjust as needed

                # Replay the frames in the buffer with the transition
                for index, frame in enumerate(buffer):

                    _, _ = cap.read()  # Read the next frame from the stream
                    if index < num_transition_frames:  # Transition period
                        alpha = 1 - (index / num_transition_frames)  # This line is modified to go from 1 to 0
                        frame = overlay_transition(frame, gradient_img, alpha)

                    if index >= num_transition_frames / 2:  # for the first 3 seconds post-transition
                        frame = overlay_replay_banner(frame)


                    cv2.imshow('Frame', frame)
                    #out_clip.write(frame)

                    duration = int(1000 / frame_rate)
                    #print(duration)  # Debugging
                    if cv2.waitKey(duration) & 0xFF == ord('q'):
                        break

                save_video(list(buffer), filename,frame_rate, frame_width, frame_height)
                # p.join()
                # p.terminate()

                buffer.clear()
                #out_clip.release()

            # If not in replay mode, write the frame to full_game.avi
            else:
                out_full.write(frame)

            if key == BACKSPACE_KEY_MAC or key == BACKSPACE_KEY:
                print("undo")
                undo_last_action(action_log)

            # Check for key press
            elif key in ACTIONS:
                current_action = ACTIONS[key]
                team = current_action['team']
                player = current_action['player']
                action = current_action['action']

                # Increment the respective action for the player
                team_info[team][player][action] += 1

                # Add to the log
                action_log.append((team, player, action))



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