from ultralytics import YOLO

class PhotoDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, image_path):
        results = self.model(image_path)
        return results
