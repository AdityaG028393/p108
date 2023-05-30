import cv2

def detect_hand_gesture(video_file):
    # Load the video file
    cap = cv2.VideoCapture(video_file)

    # Load the Haar cascade for hand detection
    palm_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "palm.xml")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale for hand detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect hands in the frame
        palms = palm_cascade.detectMultiScale(gray, 1.3, 5)

        # Draw rectangles around the detected hands
        for (x, y, w, h) in palms:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Hand Gesture Detection', frame)

        # Check for thumbs up or thumbs down gestures
        if len(palms) > 0:
            for (x, y, w, h) in palms:
                hand_roi = gray[y:y+h, x:x+w]
                _, hand_binary = cv2.threshold(hand_roi, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(hand_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 10000:
                        cv2.drawContours(frame, [contour+(x, y)], -1, (255, 0, 0), 3)
     
                        hull = cv2.convexHull(contour)
                        defects = cv2.convexityDefects(contour, cv2.convexHull(contour, returnPoints=False))

                        if defects is not None and len(defects) > 0:
                            thumbs_down = 0
                            for i in range(defects.shape[0]):
                                s, e, f, d = defects[i, 0]
                                start = tuple(contour[s][0])
                                end = tuple(contour[e][0])
                                far = tuple(contour[f][0])

                                angle = cv2.fastAtan2(end[1] - start[1], end[0] - start[0])
                                if angle < 0:
                                    angle += 360

                                if angle > 60 and angle < 120 and d > 10000:
                                    thumbs_down += 1

                            if thumbs_down > 0:
                                print("Thumbs Down: Your friend doesn't like the dress.")
                            else:
                                print("Thumbs Up: Your friend likes the dress.")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
video_file = "friend_gesture.mp4" 

detect_hand_gesture(video_file)
