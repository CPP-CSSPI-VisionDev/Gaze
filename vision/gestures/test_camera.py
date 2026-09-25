import cv2  # opencv: handles the camera and drawing

cap = cv2.VideoCapture(0)   # 0 = your built-in webcam

while True:
    ok, frame = cap.read()      # grab one picture from the camera
    if not ok:
        break
    cv2.imshow("Gaze camera test", frame)   # show it in a window
    if cv2.waitKey(1) == ord("q"):          # press q to quit
        break

cap.release()
cv2.destroyAllWindows()
