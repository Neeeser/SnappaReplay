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

        frame = receive_surface(frame)
        pygame_surface_array = pygame.surfarray.array3d(frame)
        frame = cv2.cvtColor(np.transpose(pygame_surface_array, (1, 0, 2)), cv2.COLOR_RGB2BGR)
        frame = resize_frame(frame, frame_size[0], frame_size[1])

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