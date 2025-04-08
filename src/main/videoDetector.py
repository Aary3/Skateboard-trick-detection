from ultralytics import YOLO

class VideoDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, video_path):                           #video path can also be a youtube link
        results = self.model.track(video_path, classes=[0, 36], show=True)  # 0: person, 36: skateboard
        return results