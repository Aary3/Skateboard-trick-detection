from ultralytics import YOLO
import pandas as pd
import torch

class VideoDetector:
    kList = [f"kx{i//2+1}" if i%2==0 else f"ky{i//2+1}" for i in range(34)]
    poseConfidenceList = [f"poseConfidence{i+1}" for i in range(17)]

    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.classes = [0, 36]                               # 0: person, 36: skateboard

    def setModel(self, model_path):
        self.model = YOLO(model_path)                               #set model path

    def setClassesToTrack(self, classes):
        self.classes = classes                                   #set classes to track

    def detect(self, video_path):                           #video path can also be a youtube link
        results = self.model.track(video_path, classes=self.classes, show=False, verbose=True, stream=True)
        return results
    
    def createPoseDataFrame(self, results):
        poseDataFrame = pd.DataFrame([tuple(1 for i in range(0,51))], columns=self.kList+self.poseConfidenceList)          #create empty dataframe with 34 columns for keypoints and 17 columns for pose confidence
        for result in results:
            if ((result.keypoints.xy is None) or (result.keypoints.conf is None)):
                poseDataFrame.loc[len(poseDataFrame)] = None
                continue
            keypoints = result.keypoints.xy.cpu().numpy()
            tempDataFrame = pd.DataFrame(keypoints[0].reshape(1,34), columns=self.kList)          #reshape keypoints to 1 row and 34 columns
            poseConfDataFrame = pd.DataFrame(result.keypoints.conf.cpu().numpy()[0].reshape(1,17), columns=self.poseConfidenceList)          #create dataframe for pose confidence
            tempDataFrame = pd.concat([tempDataFrame, poseConfDataFrame], axis=1)          #concatenate keypoints and pose confidence dataframes
            poseDataFrame = pd.concat([poseDataFrame, tempDataFrame])          #append keypoints to dataframe
        poseDataFrame = poseDataFrame.iloc[1:]
        poseDataFrame.reset_index(drop=True, inplace=True)
        return poseDataFrame
    
    def createPositionDataFrame(self, results):
        frame = 1
        for result in results:
            if frame == 1:
                if(result.boxes.xyxy.cpu().numpy().size == 0):
                    positionDataFrame = pd.DataFrame([(1,1,1,1,1)], columns=["px1", "py1", "px2", "py2", "positionConfidence"])
                    positionDataFrame.loc[len(positionDataFrame)] = None
                    positionDataFrame = positionDataFrame.iloc[1:]
                    positionDataFrame.reset_index(drop=True, inplace=True)
                else:
                    positionDataFrame = pd.DataFrame(result.boxes.xyxy.cpu().numpy()[0].reshape(1,4), columns=["px1", "py1", "px2", "py2"])
                    positionDataFrame["positionConfidence"] = result.boxes.conf.cpu().numpy()[0].reshape(1,1)
            else:
                if(result.boxes.xyxy.cpu().numpy().size == 0):
                    positionDataFrame.loc[len(positionDataFrame)] = None
                    frame += 1
                    continue
                if(result.boxes.xyxy.cpu().numpy().size > 4):
                    tempDataFrame = pd.DataFrame(result.boxes.xyxy.cpu().numpy()[0].reshape(1,4), columns=["px1", "py1", "px2", "py2"])
                    tempDataFrame["positionConfidence"] = result.boxes.conf.cpu().numpy()[0].reshape(1,1)
                else:
                    tempDataFrame = pd.DataFrame(result.boxes.xyxy.cpu().numpy().reshape(1,4), columns=["px1", "py1", "px2", "py2"])
                    tempDataFrame["positionConfidence"] = result.boxes.conf.cpu().numpy().reshape(1,1)
                positionDataFrame = pd.concat([positionDataFrame, tempDataFrame], axis=0)
            frame += 1
        positionDataFrame = positionDataFrame.reset_index(drop=True)
        return positionDataFrame

    def createFullDataFrame(self, positionDataFrame, poseDataFrame):
        fullDataFrame = pd.concat([positionDataFrame, poseDataFrame], axis=1)          #concatenate position and pose dataframes
        fullDataFrame = fullDataFrame.reset_index(drop=True)
        return fullDataFrame
    
    def cleanUpFullDataFrame(self, fullDataFrame):
        fullDataFrame.drop(columns=["positionConfidence"]+self.poseConfidenceList, inplace=True)            #remove confidence columns
        fullDataFrame = fullDataFrame.reset_index(drop=True)
        return fullDataFrame
