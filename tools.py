import os

import cv2

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



def video_writer(queue, output_filename, frame_size, fps):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_filename, fourcc, fps, frame_size)

    while True:
        frame = queue.get()
        if frame is None:  # Use None as a sentinel to exit the loop
            break
        out.write(frame)
    out.release()
    print(f"Saved to {output_filename}")