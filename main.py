import cv2
import mediapipe as mp
import numpy as np

# mediapipe lines to draw the tracking point on palm of hand
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence = 0.7, min_tracking_confidence = 0.7)
mp_draw = mp.solutions.drawing_utils

draw_color = (255, 255, 255)
canvas = None
drawing = False
last_x, last_y = None, None
wCam, hCam = 1080, 720
button_state = False

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1) #just fliping the image
    h, w, _ = frame.shape #extracting the dimenssions of the image

    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8) #making a 0 matrix

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #converting to RGB for mediapipe
    results = hands.process(rgb_frame) #process method takes an RGB image as input and runs the hand detection model on it

    cv2.circle(frame, (200, 100), 30, (0, 255, 0), cv2.FILLED) #drawing green dot

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            #MediaPipe's landmark.x and landmark.y are given as values between 0 and 1.
            #To match the actual size of the frame in pixels, you multiply:
            # landmark.x * w gives the pixel location for x.
            # landmark.y * h gives the pixel location for y.
            # This converts the normalized values into pixel-based coordinates that can be used directly on the frame.
            x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w)
            y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h)

            if last_x is not None and last_y is not None:
                cv2.line(canvas, (last_x, last_y), (x, y), draw_color, 5)

            last_x, last_y = x, y

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    else:
        #if no hand is detected then reseting back to NONE
        last_x, last_y = None, None

    #OpenCV’s addWeighted() function to overlay the canvas (where the drawing occurs) onto the frame (the live video feed) to create a combined frame with both the original video and the drawn lines.
    combine_frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    cv2.imshow("Air draw", combine_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 