import os
import sys
import time

import cv2
import numpy as np
import pygame

root_video_path = 'videos/'


def save_video(buffer, filename, frame_rate, frame_width, frame_height, fourcc=cv2.VideoWriter_fourcc(*'XVID')) -> None:
    print(len(buffer)/frame_rate)
    # Check if those directory exists, if not create them
    if not os.path.exists(root_video_path):
        os.makedirs(root_video_path)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    out_clip = cv2.VideoWriter(filename, fourcc, frame_rate, (frame_width, frame_height))
    for frame in buffer:
        out_clip.write(frame)


    out_clip.release()
    print(f"Saved to {filename}")
    sys.exit(0)


def video_writer(queue, output_filename, frame_size, fps):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_filename, fourcc, fps, frame_size)

    while True:
        frame = queue.get()
        if frame is None:  # Use None as a sentinel to exit the loop
            break

        # frame = receive_surface(frame)
        # pygame_surface_array = pygame.surfarray.array3d(frame)
        # frame = cv2.cvtColor(np.transpose(pygame_surface_array, (1, 0, 2)), cv2.COLOR_RGB2BGR)
        # # frame = resize_frame(frame, frame_size[0], frame_size[1])
        # if frame.shape[:2] != frame_size:
        #     frame = resize_frame(frame, frame_size[0], frame_size[1])
        out.write(frame)

    out.release()
    print(f"Saved to {output_filename}")


def save_video_process(queue, filename, frame_rate, frame_width, frame_height, fourcc=cv2.VideoWriter_fourcc(*'XVID')) -> None:
    while True:
        buffer = queue.get()

        if buffer is None:  # Use None as a sentinel to exit the loop
            break


        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = filename + f'saved_clip_{timestamp}.avi'
        print(len(buffer)/frame_rate)
        # Check if those directory exists, if not create them
        if not os.path.exists(root_video_path):
            os.makedirs(root_video_path)
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        out_clip = cv2.VideoWriter(filename, fourcc, frame_rate, (frame_width, frame_height))
        for frame in buffer:
            frame = receive_surface(frame)
            pygame_surface_array = pygame.surfarray.array3d(frame)
            frame = cv2.cvtColor(np.transpose(pygame_surface_array, (1, 0, 2)), cv2.COLOR_RGB2BGR)
            frame = resize_frame(frame, frame_width,frame_height)

            out_clip.write(frame)


        out_clip.release()

        print(f"Saved to {filename}")


def send_surface(surface):
    surface_str = pygame.image.tostring(surface, "RGB")
    width, height = surface.get_size()
    return {"data": surface_str, "size": (width, height), "format": "RGB"}


def receive_surface(data):
    if not isinstance(data, dict):
        raise TypeError(f"Expected a dictionary but got {type(data)}")

    return pygame.image.fromstring(data["data"], data["size"], data["format"])


def resize_frame(frame, width, height):
    aspect_ratio = width / height

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


def resize_surface(surface, width, height):
    # The original dimensions of the surface
    orig_width, orig_height = surface.get_size()

    aspect_ratio = orig_width / orig_height

    # Calculate the new dimensions preserving aspect ratio
    if width / height < aspect_ratio:  # Screen is "shorter" in aspect ratio than video
        new_width = width
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = height
        new_width = int(new_height * aspect_ratio)

    # Resize the surface
    resized_surface = pygame.transform.scale(surface, (new_width, new_height))

    # Create a new black surface (background)
    background = pygame.Surface((width, height))
    background.fill((0, 0, 0))

    # Calculate offsets to center the resized_surface on the background
    y_offset = (height - new_height) // 2
    x_offset = (width - new_width) // 2

    # Blit the resized_surface onto the background
    background.blit(resized_surface, (x_offset, y_offset))

    return background


def draw_scoreboard(frame, team_info, elapsed_time, frame_width, frame_height):
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