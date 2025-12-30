import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

# Get ACTUAL dimensions from camera
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera resolution: {width}x{height}")

cv2.namedWindow("Hand Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Hand Detection", 640, 480)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("test.mp4", fourcc, 30, (width, height))

n = 0
while n < 100:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        out.write(frame)  # Save frame
        n += 1
        print(f"Captured frame {n}")

    cv2.imshow("Hand Detection", frame)  # Same window name!
    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()  # Don't forget this!
cv2.destroyAllWindows()
