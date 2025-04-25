from ultraytics import YOLO
import torch
from videoDetector import VideoDetector

#model = YOLO('C:/Users/piotr/yolov12-main/yolo11n-pose.pt')
#print(f"Model is running on: {model.device}")

#print("CUDA Available:", torch.cuda.is_available())  # Should print True
#print("Current Device:", torch.cuda.current_device())  # Should print 0 (or another valid GPU index)
#print("Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")

# Train the model
#results = model.train(
#  data='coco128.yaml',
#  epochs=50, 
#  batch=80, 
#  imgsz=640,
#  scale=0.5,  # S:0.9; M:0.9; L:0.9; X:0.9
#  mosaic=1.0,
#  mixup=0.0,  # S:0.05; M:0.15; L:0.15; X:0.2
#  copy_paste=0.1,  # S:0.15; M:0.4; L:0.5; X:0.6
#  workers=2,
#  device="cuda:0",
#)

class YOLOtrainer:
    def __init__(self, videos):
        self.videos=videos

    def createTrainingSet(self):
        for video in self.videos:
          # Create a VideoDetector instance
          detector = VideoDetector(model_path="models/yolo/COCO pretrained/yolo11n-pose.pt")
          detector.setClassesToTrack([0])                               # 0: person
          # Perform detection on the video
          results = detector.detect(video)
          poseDataFrame = detector.createPoseDataFrame(results)          #create pose dataframe

          detector.setModel("models/yolo/COCO pretrained/yolo11n.pt")
          detector.setClassesToTrack([36])                               # 36: skateboard
          results = detector.detect(video)                                #detect again with normal model
          positionDataFrame = detector.createPositionDataFrame(results)

          fullDataFrame = detector.createFullDataFrame(positionDataFrame, poseDataFrame)          #create full dataframe
          fullDataFrame = detector.cleanUpFullDataFrame(fullDataFrame)          #clean up full dataframe
          videoTensor = detector.createTensor(fullDataFrame)
          trickName = "ollie"

          TrainingSet.append(videoTensor, trickName)