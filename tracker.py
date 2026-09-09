from ultralytics import YOLO


class FootballTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def track(self, frame):

        results = self.model.track(
            frame,
            persist=True,
            tracker="trackers/botsort.yaml",
            verbose=False
        )

        return results