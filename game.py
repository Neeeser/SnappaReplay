import os, cv2, time
import queue
import sys
from collections import deque
from multiprocessing import Process, Queue
import numpy as np
import pygame
from tools import video_writer, send_surface, receive_surface, resize_surface, save_video_process

## Define keyboard Macros as sets
TEAM1_PLAYER1_HIT = {'w'}
TEAM1_PLAYER1_SINK = {'e'}
TEAM1_PLAYER1_MISS = {'r'}
TEAM1_PLAYER1_DROP = {'t'}

TEAM1_PLAYER2_HIT = {'z'}
TEAM1_PLAYER2_SINK = {'x'}
TEAM1_PLAYER2_MISS = {'c'}
TEAM1_PLAYER2_DROP = {'v'}

TEAM2_PLAYER1_HIT = {'u'}
TEAM2_PLAYER1_SINK = {'i'}
TEAM2_PLAYER1_MISS = {'o'}
TEAM2_PLAYER1_DROP = {'p'}

TEAM2_PLAYER2_HIT = {'n'}
TEAM2_PLAYER2_SINK = {'m'}
TEAM2_PLAYER2_MISS = {','}
TEAM2_PLAYER2_DROP = {'.'}


SPEED_UP = {'='}
SPEED_DOWN = {'-'}
REPLAY ={'s'}
BACKSPACE_KEY = ord('\x08')  # or simply 8
BACKSPACE_KEY_MAC = 127

Quit = {'escape'}


andrew_url = "http://admin:4647@andrew.local:8081/video"

class SnappaWindow:
    def __init__(self, stream_url=None, team_info=None, debug=False):
        # All your initialization code...
        self.stream_url = stream_url

        # Initialize the camera
        self.cap = self.initialize_camera(self.stream_url)
        if self.cap:
            self.frame_width = int(self.cap.get(3))
            self.frame_height = int(self.cap.get(4))
        else:
            self.frame_width = 640
            self.frame_height = 480

        self.screen = pygame.display.set_mode((self.frame_width, self.frame_height), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()


        # Add other initializations as necessary...
        # Path where videos are stored
        self.root_video_path = 'videos/'
        # Timestamp for the video
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        # Filename for the video
        self.filename = f'Game_{self.timestamp}/'
        self.full_video_path = self.root_video_path + self.filename
        self.full_video_name = self.full_video_path + 'full_game.avi'

        # Check if those directory exists, if not create them
        if not os.path.exists(self.root_video_path):
            os.makedirs(self.root_video_path)
        if not os.path.exists(self.full_video_path):
            os.makedirs(self.full_video_path)

        # Define the codec and create VideoWriter object
        # self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS) or 30  # default to 30 FPS if not provided
        # self.out_full = cv2.VideoWriter(self.full_video_name, self.fourcc, self.frame_rate,
        #                                 (self.frame_width, self.frame_height))

        self.slow_mo_ratio = 3
        self.replay_index = 0
        print(self.frame_rate)

        # Create a queue for the video writer
        self.output_queue = Queue()

        self.writer_process = Process(target=video_writer, args=(self.output_queue, self.full_video_name,  (self.frame_width, self.frame_height), self.frame_rate))
        self.writer_process.start()

        # Create a queue for the save video writer
        self.clip_output_queue = Queue()
        self.clip_output_process = Process(target=save_video_process, args=(self.clip_output_queue, self.full_video_path, self.frame_rate, self.frame_width, self.frame_height))
        self.clip_output_process.start()

        # Create a queue for the clip_buffer
        self.replay_duration = 5
        self.clip_buffer = deque(maxlen=int(self.frame_rate) * self.replay_duration)
        self.replay_mode = False
        self.clip_buffer_size = 0


        # Time and team
        self.elapsed_time = 0
        if not team_info:
            team_info = {
                'Team1': {
                    'TeamName': "Messy Room",
                    'PlayerOne': {'PlayerName': "Trey", 'Hits': 0, 'Misses': 0, 'Sinks': 0, 'Drops': 0, 'Stats': 0.0},
                    'PlayerTwo': {'PlayerName': "James", 'Hits': 0, 'Misses': 0, 'Sinks': 0, 'Drops': 0, 'Stats': 0.0},
                    'TeamPoints': 0
                },
                'Team2': {
                    'TeamName': "Ragno Club",
                    'PlayerOne': {'PlayerName': "Ben", 'Hits': 0, 'Misses': 0, 'Sinks': 0, 'Drops': 0, 'Stats': 0.0},
                    'PlayerTwo': {'PlayerName': "Colin", 'Hits': 0, 'Misses': 0, 'Sinks': 0, 'Drops': 0, 'Stats': 0.0},
                    'TeamPoints': 0
                }
            }

        self.team_info = team_info

        pygame.font.init()
        self.font = pygame.font.Font(None, 36)  # Font for FPS display

        self.screen_info = pygame.display.Info()
        self.screen_width = self.screen_info.current_w
        self.screen_height = self.screen_info.current_h
        print(f"Screen width: {self.screen_width}, Screen height: {self.screen_height}")



        self.debug = debug
    def initialize_camera(self, url):
        if url:
            for i in range(5):
                cap = cv2.VideoCapture(url)
                if cap.isOpened():
                    return cap
                else:
                    time.sleep(1)
                    print(f"Failed to connect to camera, retrying {i + 1}/5")

        print("Trying built in camera")
        cap = cv2.VideoCapture(0)  # 0 for the default camera, change if you have multiple cameras
        if cap.isOpened():
            return cap

        return None


    def draw_scoreboard_pygame(self, team_info, elapsed_time):
        w, h = self.screen.get_size()

        # Define properties of the scoreboard
        height = h // 8
        color = (155, 50, 42)

        # Create a semi-transparent rectangle (overlay) for the scoreboard
        overlay = pygame.Surface((w, height))
        overlay.set_alpha(180)  # Alpha value
        overlay.fill(color)
        self.screen.blit(overlay, (0, h - height))

        # Define font and colors
        font_size = int(min(w/1.2, h) / 30)  # Adjust based on the look
        font = pygame.font.Font(pygame.font.get_default_font(), font_size)
        header_color = (255, 255, 255)
        text_color = (168, 224, 58)

        left_position = int(w * 0.01)
        right_position = int(w * 0.55)

        # Draw Team's info for both teams
        for team, data in team_info.items():
            vertical_position = h - height + font_size / 10  # Reset the vertical position for each team
            # Team's Name and Points
            team_text = f"{data['TeamPoints']} Points - {data['TeamName']}"

            if team == 'Team1':
                team_text = f"{data['TeamName']} - {data['TeamPoints']} Points"

            text_surface = font.render(team_text, True, header_color)

            if team == 'Team1':
                position = (left_position, vertical_position)
            else:
                position = (w - text_surface.get_width() - left_position, vertical_position)

            self.screen.blit(text_surface, position)
            vertical_position += font_size * 1.2  # Adjust for spacing

            # Draw Players' stats
            for player in ['PlayerOne', 'PlayerTwo']:
                player_info = data[player]
                formatted_stats = "{:+.2f}".format(player_info["Stats"])
                player_text = f"{player_info['PlayerName']} | {formatted_stats} H: {player_info['Hits']} M: {player_info['Misses']} S: {player_info['Sinks']} D: {player_info['Drops']}"
                text_surface = font.render(player_text, True, text_color)

                if team == 'Team1':
                    player_text = f"H: {player_info['Hits']} M: {player_info['Misses']} S: {player_info['Sinks']} D: {player_info['Drops']} {formatted_stats} | {player_info['PlayerName']}"
                    position = (left_position, vertical_position)
                    text_surface = font.render(player_text, True, text_color)

                else:
                    position = (w - text_surface.get_width() - left_position, vertical_position)


                self.screen.blit(text_surface, position)
                vertical_position += font_size * 1.2

        # Display elapsed time at the bottom center
        time_surface = font.render(elapsed_time, True, header_color)
        time_position = (w // 2 - time_surface.get_width() // 2, h - font_size)
        self.screen.blit(time_surface, time_position)

    def resize_frame(self, frame):
        aspect_ratio = self.frame_width / self.frame_height
        width = self.screen_width
        height = self.screen_height
        # Calculate the new dimensions preserving aspect ratio
        if width / height < aspect_ratio:  # Screen is "shorter" in aspect ratio than video
            new_width = width
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = height
            new_width = int(new_height * aspect_ratio)

        # Resize the frame to the new dimensions
        frame_resized = cv2.resize(frame, (new_width, new_height))

        # Create a black background
        background = np.zeros((height, width, 3), dtype=np.uint8)

        # Overlay the resized frame onto the black background
        y_offset = (height - new_height) // 2
        x_offset = (width - new_width) // 2
        background[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = frame_resized

        return background



    def process_frame_cv2(self, frame):
        # All processing on the frame...
        # Orentation, cropping, etc...
        frame = cv2.flip(frame, 1)

        # Like drawing the scoreboard and any overlays...
        #frame = self.draw_scoreboard(frame, self.team_info, self.elapsed_time)


        # Resize the frame to fit the screen
        frame = self.resize_frame(frame)



        return frame



    def process_frame_pygame(self):
        # All processing on the frame...
        self.draw_scoreboard_pygame(self.team_info, self.elapsed_time)

    def overlay_replay_banner(self, frame_surface: pygame.Surface, overlay_width_percentage=0.3):
        # Define properties of the replay banner
        overlay_path = "imgs/ReplayOverlay.png"

        # Desired width of the overlay as a percentage of frame's width
        frame_width, frame_height = frame_surface.get_size()
        desired_width = int(frame_width * overlay_width_percentage)

        # Load the overlay image (assuming it has a transparent background)
        overlay = pygame.image.load(overlay_path).convert_alpha()
        original_width, original_height = overlay.get_size()

        # Calculate the scale factor
        scale = desired_width / original_width

        # Resize the overlay image
        overlay = pygame.transform.scale(overlay, (int(original_width * scale), int(original_height * scale)))
        overlay_width, overlay_height = overlay.get_size()

        # Calculate the position to center the overlay on the frame
        x = (frame_width - overlay_width) // 2
        y = 0

        # Blit the overlay onto the frame surface
        frame_surface.blit(overlay, (x, y))


        # Calculate and format the decimal speed
        decimal_speed = 1 / self.slow_mo_ratio
        speed_text = f"{decimal_speed:.2f}"  # This will format the number to two decimal places

        font = pygame.font.SysFont(pygame.font.get_default_font(), 36)  # Choose appropriate font and size
        text_surface = font.render(f"Speed: {speed_text}x", True, (255, 255, 255))  # White color for text
        text_width, text_height = text_surface.get_size()
        frame_surface.blit(text_surface, (frame_width - text_width - 10, 10))  # 10 pixel margin from top right


        return frame_surface


    def process_replay_frame_pygame(self, frame):
        width = self.screen_width
        height = self.screen_height

        b = resize_surface(frame, width, height)
        self.screen.fill((0, 0, 0))  # fill the surface with the color black
        overlay = self.overlay_replay_banner(b)
        self.screen.blit(overlay, (0, 0))

    def mainloop(self):
        try:
            running = True
            start_time = time.time()

            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    # Add other events like keypresses...
                    elif event.type == pygame.VIDEORESIZE:
                        self.screen_width = event.w
                        self.screen_height = event.h
                        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.RESIZABLE)


                    # Check if key Quit was pressed
                    elif event.type == pygame.KEYDOWN:
                        key = pygame.key.name(event.key)
                        if key in Quit:
                            running = False

                        elif key in REPLAY:
                            print("Replay mode")
                            self.replay_mode = True

                        elif key in SPEED_UP:
                            if self.slow_mo_ratio > 1:
                                self.slow_mo_ratio -= 1


                        elif key in SPEED_DOWN:
                            self.slow_mo_ratio += 1
                            if self.slow_mo_ratio == 0:
                                self.slow_mo_ratio += 1

                ret, frame = self.cap.read()

                if not ret:
                    print("Failed to grab frame")
                    break

                # Get the time elapsed
                elapsed_time_sec = int(time.time() - start_time)
                self.elapsed_time = f"{elapsed_time_sec // 60:02d}:{elapsed_time_sec % 60:02d}"  # Convert to MM:SS format

                # Draw the scoreboard
                frame = self.process_frame_cv2(frame)

                # Convert to RGB and transpose
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb_transposed = np.transpose(frame_rgb, (1, 0, 2))
                frame_surface = pygame.surfarray.make_surface(frame_rgb_transposed)

                # Display the frame
                self.screen.blit(frame_surface, (0, 0))
                self.process_frame_pygame()
                self.output_queue.put(send_surface(pygame.display.get_surface()))

                # 3. Put the Frame in the Buffer for the Video Writer

                if self.replay_mode:
                    try:

                        frame = receive_surface(self.clip_buffer[self.replay_index])
                        print(self.clock.get_fps())
                        self.process_replay_frame_pygame(frame)
                        self.replay_index += 1

                        # If you've reached the end of the buffer
                        if self.replay_index >= len(self.clip_buffer):
                            self.replay_mode = False
                            self.replay_index = 0
                            self.clip_output_queue.put(list(self.clip_buffer))
                            self.clip_buffer.clear()

                    except IndexError:
                        self.replay_mode = False
                        self.replay_index = 0


                else:

                    self.clip_buffer.append(send_surface(pygame.display.get_surface()))

                if self.debug:
                    # Display FPS
                    fps = int(self.clock.get_fps())
                    fps_text = self.font.render(f"FPS: {fps}", True, (255, 255, 255))

                    # Set the coordinates to (0, 0) to position in the top-left corner
                    self.screen.blit(fps_text, (0, 0))

                # Update the display
                pygame.display.flip()

                self.clock.tick(self.frame_rate + 1 if not self.replay_mode else self.frame_rate / self.slow_mo_ratio)







        except KeyboardInterrupt:
            pass

        except Exception as e:
            print(e)

        finally:
            print("Cleaning up")
            self.cleanup()

    def cleanup(self):
        print("Initiating cleanup...")
        self.output_queue.put(None)
        self.clip_output_queue.put(None)


        self.cap.release()
        print("Camera released")
        pygame.quit()
        print("Pygame quit")

        self.writer_process.join()
        print("Writer process joined")
        self.writer_process.terminate()

        # Close the queues properly
        self.output_queue.close()
        self.clip_output_queue.close()
        print("Queues closed")


        sys.exit()



if __name__ == "__main__":

    game = SnappaWindow(debug=True)
    game.mainloop()