from ultralytics import YOLO
import pandas as pd
import numpy as np
import torch

class VideoDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.classes = [0, 36]                               # 0: person, 36: skateboard

    def setModel(self, model_path):
        self.model = YOLO(model_path)                               #set model path

    def setClassesToTrack(self, classes):
        self.classes = classes                                   #set classes to track

    def detect(self, video_path):                           #video path can also be a youtube link
        results = self.model.track(video_path, classes=self.classes, show=False, verbose=False, stream=True)  # 0: person, 36: skateboard
        return results
    
    def createPoseDataFrame(self, results):
        poseDataFrame = pd.DataFrame([(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)], columns=["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4", "x5", "y5", "x6", "y6", "x7", "y7", "x8", "y8", "x9", "y9", "x10", "y10", "x11", "y11", "x12", "y12", "x13", "y13", "x14", "y14", "x15", "y15", "x16", "y16", "x17", "y17"])
        for result in results:
            if ((result.keypoints.xy is None) or (result.keypoints.conf is None)):
                poseDataFrame.loc[len(poseDataFrame)] = None
                continue
            keypoints = result.keypoints.xy.cpu().numpy()
            #keypointsconf = result.keypoints.conf.cpu().numpy()
            tempDataFrame = pd.DataFrame(keypoints[0].reshape(1,34), columns=["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4", "x5", "y5", "x6", "y6", "x7", "y7", "x8", "y8", "x9", "y9", "x10", "y10", "x11", "y11", "x12", "y12", "x13", "y13", "x14", "y14", "x15", "y15", "x16", "y16", "x17", "y17"])          #reshape keypoints to 1 row and 34 columns
            poseDataFrame = pd.concat([poseDataFrame, tempDataFrame])          #append keypoints to dataframe
        print(poseDataFrame)
        poseDataFrame = poseDataFrame.iloc[1:]         #remove first row of dataframe
        poseDataFrame.reset_index(drop=True, inplace=True)          #reset index of dataframe
        print(poseDataFrame)
        return poseDataFrame
    
    def createPositionDataFrame(self, results):
        frame = 1
        for result in results:
            #print(result.boxes.xyxy.cpu().numpy())
            if frame == 1:
                positionDataFrame = pd.DataFrame(result.boxes.xyxy.cpu().numpy(), columns=["x1", "y1", "x2", "y2"])
                #positionDataFrame["positionConfidence"] = result.boxes.conf.cpu().numpy()
                #positionDataFrame["class"] = result.boxes.cls.cpu().numpy()
            else:
                #print(result.boxes.xyxy.cpu().numpy())
                if(result.boxes.xyxy.cpu().numpy().size == 0):
                    positionDataFrame.loc[len(positionDataFrame)] = None
                    frame += 1
                    continue
                if(result.boxes.xyxy.cpu().numpy().size > 4):
                    tempDataFrame = pd.DataFrame(result.boxes.xyxy.cpu().numpy()[0].reshape(1,4), columns=["x1", "y1", "x2", "y2"])
                else:
                    tempDataFrame = pd.DataFrame(result.boxes.xyxy.cpu().numpy().reshape(1,4), columns=["x1", "y1", "x2", "y2"])
                #tempDataFrame["positionConfidence"] = result.boxes.conf.cpu().numpy()
                #tempDataFrame["class"] = result.boxes.cls.cpu().numpy()
                positionDataFrame = pd.concat([positionDataFrame, tempDataFrame], axis=0)
            frame += 1
        positionDataFrame = positionDataFrame.reset_index(drop=True)
        #positionDataFrame.drop(positionDataFrame[positionDataFrame['confidence'] < 0.5].index, inplace=True)            #remove rows with confidence < 0.5
        #positionDataFrame = positionDataFrame.reset_index(drop=True)
        #positionDataFrame = positionDataFrame.drop(columns=["confidence"])
        #print(positionDataFrame)
        #positionTensor = torch.tensor(positionDataFrame.values, dtype=torch.float32)
        #print(positionTensor)
        print(positionDataFrame)
        return positionDataFrame

    def createFullDataFrame(self, positionDataFrame, poseDataFrame):
        fullDataFrame = pd.concat([positionDataFrame, poseDataFrame], axis=1)          #concatenate position and pose dataframes
        fullDataFrame = fullDataFrame.reset_index(drop=True)
        return fullDataFrame
    
    def cleanUpFullDataFrame(self, fullDataFrame):
        fullDataFrame.drop(fullDataFrame[fullDataFrame['positionConfidence'] < 0.5].index, inplace=True)            #remove rows with confidence < 0.5
        fullDataFrame.drop(fullDataFrame[fullDataFrame['poseConfidence'] < 0.5].index, inplace=True)            #remove rows with confidence < 0.5
        fullDataFrame.drop(columns=["positionConfidence", "poseConfidence"], inplace=True)            #remove confidence columns
        fullDataFrame = fullDataFrame.reset_index(drop=True)
        print(fullDataFrame)
        return fullDataFrame
    
    def createFinalDataFrame(self, fullDataFrame):
        positionDataFrame = fullDataFrame.iloc[:, :4]          #position data is in the first 4 columns
        positionDataFrame.dropna(inplace=True, how='any')            #remove rows with NaN values
        positionDataFrame = positionDataFrame.reset_index(drop=True)
        poseDataFrame = fullDataFrame.iloc[:, 4:]              #pose data is in the last 4 columns
        poseDataFrame.dropna(inplace=True, how='any')            #remove rows with NaN values
        poseDataFrame = poseDataFrame.reset_index(drop=True)

        positionTensor = torch.tensor(positionDataFrame.values, dtype=torch.float32)
        poseTensor = torch.tensor(poseDataFrame.values, dtype=torch.float32)

        #finalDataFrame = pd.DataFrame([positionTensor, poseTensor], columns=["position", "pose"])
        #finalDataFrame = finalDataFrame.reset_index(drop=True)
        #return finalDataFrame
