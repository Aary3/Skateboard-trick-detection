from ultralytics import YOLO

class PhotoDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, image_path):
        results = self.model(image_path, classes=[0, 36])           # 0: person, 36: skateboard
        return results
