import cv2
import numpy as np
import main
# Initialize variables
active_field = None
team1_name = ''
team2_name = ''

def on_mouse_click(event, x, y, flags, param):
    global active_field
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if a particular input field was clicked
        if team1_rect[0] < x < team1_rect[2] and team1_rect[1] < y < team1_rect[3]:
            active_field = 'team1'
        elif team2_rect[0] < x < team2_rect[2] and team2_rect[1] < y < team2_rect[3]:
            active_field = 'team2'
        else:
            active_field = None

# Define input field coordinates
team1_rect = [50, 100, 450, 150]  # [x1, y1, x2, y2]
team2_rect = [50, 200, 450, 250]

# Create white menu frame
frame = np.ones((400, 500, 3), dtype=np.uint8) * 255

# Set mouse callback
cv2.namedWindow('Menu')
cv2.setMouseCallback('Menu', on_mouse_click)

while True:
    temp_frame = frame.copy()
    cv2.rectangle(temp_frame, (team1_rect[0], team1_rect[1]), (team1_rect[2], team1_rect[3]), (0, 0, 255), 1)
    cv2.rectangle(temp_frame, (team2_rect[0], team2_rect[1]), (team2_rect[2], team2_rect[3]), (0, 0, 255), 1)

    cv2.putText(temp_frame, team1_name, (team1_rect[0]+10, team1_rect[1]+40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(temp_frame, team2_name, (team2_rect[0]+10, team2_rect[1]+40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Menu', temp_frame)

    key = cv2.waitKey(1) & 0xFF  # Take only the last 8 bits
    valid_ascii = range(32, 127)  # This range includes common keyboard keys excluding control keys


    if key == 27:  # ESC key to exit
        break
    elif key == 13:  # Enter key to transition to video capture
        main.main()
    if key in valid_ascii:
        if active_field == 'team1':
            team1_name += chr(key)
        elif active_field == 'team2':
            team2_name += chr(key)

cv2.destroyAllWindows()
