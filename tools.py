import cv2


def save_video(buffer, filename, frame_rate, frame_width, frame_height, fourcc=cv2.VideoWriter_fourcc(*'XVID')) -> None:
    print(len(buffer)/frame_rate)
    out_clip = cv2.VideoWriter(filename, fourcc, frame_rate, (frame_width, frame_height))
    for frame in buffer:
        out_clip.write(frame)


    out_clip.release()
    print(f"Saved to {filename}")