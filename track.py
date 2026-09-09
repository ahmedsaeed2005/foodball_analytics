import cv2
from tracker import FootballTracker

# تحميل التراكر
tracker = FootballTracker("runs_players_model-5_weights_best.pt")

# فتح الفيديو
cap = cv2.VideoCapture("Ronaldo.mp4")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Tracking
    results = tracker.track(frame)

    # رسم النتائج
    annotated_frame = results[0].plot()

    cv2.imshow("Football Analytics", annotated_frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()