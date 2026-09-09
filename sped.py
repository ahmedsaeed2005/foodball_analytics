import cv2
import torch
import math
from ultralytics import YOLO

# ==========================
# اختيار الجهاز
# ==========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on:", device)

# ==========================
# تحميل الموديل
# ==========================
model = YOLO("runs_players_model-5_weights_best.pt")
model.to(device)

# ==========================
# فتح الفيديو
# ==========================
cap = cv2.VideoCapture("Ronaldo.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
dt = 1 / fps

print(f"FPS : {fps}")

# ==========================
# Dictionaries
# ==========================
previous_positions = {}
smoothed_speeds = {}

# معامل التنعيم
ALPHA = 0.3

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # ==========================
    # Tracking
    # ==========================
    results = model.track(
        frame,
        persist=True,
        tracker="trackers/botsort.yaml",
        device=device,
        conf=0.3,
        imgsz=640
    )

    annotated_frame = results[0].plot()

    boxes = results[0].boxes

    if boxes.id is not None:

        ids = boxes.id.cpu().numpy().astype(int)
        xywh = boxes.xywh.cpu().numpy()

        for box, track_id in zip(xywh, ids):

            x, y, w, h = box

            # ==========================
            # Bottom Center
            # ==========================
            foot_x = int(x)
            foot_y = int(y + h / 2)

            foot = (foot_x, foot_y)

            speed = 0

            # ==========================
            # حساب السرعة
            # ==========================
            if track_id in previous_positions:

                prev_x, prev_y = previous_positions[track_id]

                dx = foot_x - prev_x
                dy = foot_y - prev_y

                distance = math.sqrt(dx ** 2 + dy ** 2)

                current_speed = distance / dt

                # ==========================
                # EMA Smoothing
                # ==========================
                if track_id in smoothed_speeds:

                    speed = (
                        ALPHA * current_speed
                        + (1 - ALPHA) * smoothed_speeds[track_id]
                    )

                else:

                    speed = current_speed

                smoothed_speeds[track_id] = speed

            previous_positions[track_id] = foot

            # ==========================
            # رسم نقطة القدم
            # ==========================
            cv2.circle(
                annotated_frame,
                foot,
                4,
                (0, 0, 255),
                -1
            )

            # ==========================
            # كتابة السرعة
            # ==========================
            cv2.putText(
                annotated_frame,
                f"ID:{track_id}",
                (foot_x - 20, foot_y - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            cv2.putText(
                annotated_frame,
                f"{speed:.1f} px/s",
                (foot_x - 20, foot_y - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

    # ==========================
    # عرض الفيديو
    # ==========================
    cv2.imshow("Football Analytics", annotated_frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()