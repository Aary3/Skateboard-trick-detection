from ultralytics import YOLO

class VideoDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, video_path):                           #video path can alsi be a youtube link
        results = self.model.track(video_path)
        return results