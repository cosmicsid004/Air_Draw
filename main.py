import cv2
import mediapipe as mp
import numpy as np
import math

# mediapipe lines to draw the tracking point on palm of hand
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence = 0.7, min_tracking_confidence = 0.7)
mp_draw = mp.solutions.drawing_utils

#default constants
draw_color = (0, 0, 0)
canvas = None
drawing = False
last_ix, last_iy = None, None
wCam, hCam = 1080, 720

#works like toggle switch for selecting drawing color
red_toggle_state = True
green_toggle_state = True
blue_toggle_state = True
ans1 = False
ans2 = False
ans3 = False

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

    cv2.rectangle(frame, (1216, 180), (1278, 429), (200, 200, 200), 4) #main rectangle in which all colors reside
    cv2.rectangle(frame, (1218, 265), (1276, 181), (0, 0, 255), -1) #red color
    cv2.rectangle(frame, (1218, 346), (1276, 262), (0, 255, 0), -1) #green color
    cv2.rectangle(frame, (1218, 427), (1276, 343), (255, 0, 0), -1) #blue color

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            #MediaPipe's landmark.x and landmark.y are given as values between 0 and 1.
            #To match the actual size of the frame in pixels, you multiply:
            # landmark.x * w gives the pixel location for x.
            # landmark.y * h gives the pixel location for y.
            # This converts the normalized values into pixel-based coordinates that can be used directly on the frame.

            #index finger tip coordinates
            ix = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w)
            iy = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h)

            #thumb tip coordinates
            tx = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * w)
            ty = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * h)

            #middle finger tip coordinates
            mx = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * w)
            my = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * h)

            cv2.circle(frame, (tx, ty), 16, (68,214,44), 4) #highlighting the finger tips
            cv2.circle(frame, (ix, iy), 16, (68,214,44), 4) #highlighting the finger tips

            thumb_index_distance = int(math.hypot(tx - ix, ty - iy)) #calculating the distance between the thumb tip and index finger tip
            #print(thumb_index_distance)
            middle_index_distance = int(math.hypot(mx - ix, my - iy)) #calculating the distance between the middle finger tip and index finger tip
            #print(middle_index_distance)

            #calculating the distance for selecting colors
            index_red_dis = int(math.hypot(ix - 1247, iy - 223))
            index_green_dis = int(math.hypot(ix - 1247, iy - 304))
            index_blue_dis = int(math.hypot(ix - 1247, iy - 385))
            #print(index_green_dis)

            #------- color switch toggle ---------
            if index_red_dis < 50:
                if not red_toggle_state:
                    ans1 = not ans1
                    red_toggle_state = True
                else:
                    red_toggle_state = False

            if index_green_dis < 50:
                if not green_toggle_state:
                    ans2 = not ans2
                    green_toggle_state = True
                else:
                    green_toggle_state = False

            if index_blue_dis < 50:
                if not blue_toggle_state:
                    ans3 = not ans3
                    blue_toggle_state = True
                else:
                    blue_toggle_state = False
                                
            if ans1:
                #print("red selected")
                ans2 = False
                ans3 = False
                draw_color = (0, 0, 255)
            if ans2:
                #print("green selected")
                ans1 = False
                ans3 = False
                draw_color = (0, 255, 0)
            if ans3:
                #print("blue selected")
                ans1 = False
                ans2 = False
                draw_color = (255, 0, 0)
            #------- color switch toggle ---------

            #drawing
            if thumb_index_distance > 50 and last_ix is not None and last_iy is not None:
                cv2.line(canvas, (last_ix, last_iy), (ix, iy), draw_color, 5)

            #eraser
            if middle_index_distance < 40 and last_ix is not None and last_iy is not None:
                cv2.circle(frame, (mx, my), 16, (68,214,44), 4) #highlighting the finger tips
                cv2.line(canvas, (last_ix, last_iy), (ix, iy), (0, 0, 0), 10)
                
            last_ix, last_iy = ix, iy

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    else:
        #if no hand is detected then reseting back to NONE
        last_ix, last_iy = None, None

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
