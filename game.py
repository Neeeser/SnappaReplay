import os, cv2, time
from multiprocessing import Process, Queue
import numpy as np
import pygame
from tools import video_writer

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

BACKSPACE_KEY = ord('\x08')  # or simply 8
BACKSPACE_KEY_MAC = 127


andrew_url = "http://admin:4647@andrew.local:8081/video"
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

class SnappaWindow:
    def __init__(self, stream_url=None, team_info=team_info):
        # All your initialization code...
        self.stream_url = stream_url


        self.cap = self.initialize_camera(self.stream_url)
        if self.cap:
            self.frame_width = int(self.cap.get(3))
            self.frame_height = int(self.cap.get(4))
        else:
            self.frame_width = 640
            self.frame_height = 480

        self.screen = pygame.display.set_mode((self.frame_width, self.frame_height))
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

        # Create a queue for the video writer
        self.output_queue = Queue()

        self.writer_process = Process(target=video_writer, args=(self.output_queue, self.full_video_name,  (self.frame_width, self.frame_height), self.frame_rate))
        self.writer_process.start()

        print(self.frame_rate)


        # Time and team
        self.elapsed_time = 0
        self.team_info = team_info

        pygame.font.init()
        self.font = pygame.font.Font(None, 36)  # Font for FPS display


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


    def draw_scoreboard(self, frame, team_info, elapsed_time):
        h, w, _ = frame.shape

        # Flips the frame horizontally
        #frame = cv2.flip(frame, 1)

        # Define properties of the scoreboard
        height = self.frame_height // 8
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
        font_thickness = int(self.frame_width * 0.001)

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

                cv2.putText(frame, player_text, (position, vertical_position), font, font_scale, text_color,
                            font_thickness,
                            cv2.LINE_AA)
                vertical_position += int(max_line_height)

            # Reset vertical position for the second team
            vertical_position = h - height + int(max_line_height)

        # ...

        # Display elapsed time at the bottom center
        time_position = (w // 2 - int(self.frame_width * 0.05), h - int(self.frame_height * 0.01))
        cv2.putText(frame, elapsed_time, time_position, font, font_scale, header_color, font_thickness, cv2.LINE_AA)

        return frame

    def process_frame(self, frame):
        # All processing on the frame...
        # Orentation, cropping, etc...
        frame = cv2.flip(frame, 1)

        # Like drawing the scoreboard and any overlays...
        frame = self.draw_scoreboard(frame, self.team_info, self.elapsed_time)
        return frame

    def mainloop(self):
        try:
            running = True
            start_time = time.time()
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    # Add other events like keypresses...

                ret, frame = self.cap.read()

                if not ret:
                    print("Failed to grab frame")
                    break

                # Draw the scoreboard
                elapsed_time_sec = int(time.time() - start_time)
                self.elapsed_time = f"{elapsed_time_sec // 60:02d}:{elapsed_time_sec % 60:02d}"  # Convert to MM:SS format
                frame = self.process_frame(frame)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb_transposed = np.transpose(frame_rgb, (1, 0, 2))
                frame_surface = pygame.surfarray.make_surface(frame_rgb_transposed)
                self.screen.blit(frame_surface, (0, 0))

                # Display FPS
                fps = int(self.clock.get_fps())
                fps_text = self.font.render(f"FPS: {fps}", True, (255, 255, 255))
                self.screen.blit(fps_text, (self.frame_width - fps_text.get_width(), 0))

                pygame.display.flip()
                self.output_queue.put(frame)
                self.clock.tick(self.frame_rate)

        except KeyboardInterrupt:
            pass

        finally:
            print("Cleaning up")
            self.cleanup()

    def cleanup(self):
        self.cap.release()
        self.output_queue.put(None)
        self.writer_process.join()
        pygame.quit()
        # Any other cleanup...

if __name__ == "__main__":

    game = SnappaWindow()
    game.mainloop()