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
        results = self.model.track(video_path, classes=self.classes, show=True, verbose=True, stream=True)  # 0: person, 36: skateboard
        return results
    
    def createPoseDataFrame(self, results):
        poseDataFrame = pd.DataFrame([(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)], columns=["kx1", "ky1", "kx2", "ky2", "kx3", "ky3", "kx4", "ky4", "kx5", "ky5", "kx6", "ky6", "kx7", "ky7", "kx8", "ky8", "kx9", "ky9", "kx10", "ky10", "kx11", "ky11", "kx12", "ky12", "kx13", "ky13", "kx14", "ky14", "kx15", "ky15", "kx16", "ky16", "kx17", "ky17", "poseConfidence1", "poseConfidence2", "poseConfidence3", "poseConfidence4", "poseConfidence5", "poseConfidence6", "poseConfidence7", "poseConfidence8", "poseConfidence9", "poseConfidence10", "poseConfidence11", "poseConfidence12", "poseConfidence13", "poseConfidence14", "poseConfidence15", "poseConfidence16", "poseConfidence17"])
        for result in results:
            if ((result.keypoints.xy is None) or (result.keypoints.conf is None)):
                poseDataFrame.loc[len(poseDataFrame)] = None
                continue
            keypoints = result.keypoints.xy.cpu().numpy()
            #keypointsconf = result.keypoints.conf.cpu().numpy()
            tempDataFrame = pd.DataFrame(keypoints[0].reshape(1,34), columns=["kx1", "ky1", "kx2", "ky2", "kx3", "ky3", "kx4", "ky4", "kx5", "ky5", "kx6", "ky6", "kx7", "ky7", "kx8", "ky8", "kx9", "ky9", "kx10", "ky10", "kx11", "ky11", "kx12", "ky12", "kx13", "ky13", "kx14", "ky14", "kx15", "ky15", "kx16", "ky16", "kx17", "ky17"])          #reshape keypoints to 1 row and 34 columns
            poseConfDataFrame = pd.DataFrame(result.keypoints.conf.cpu().numpy()[0].reshape(1,17), columns=["poseConfidence1", "poseConfidence2", "poseConfidence3", "poseConfidence4", "poseConfidence5", "poseConfidence6", "poseConfidence7", "poseConfidence8", "poseConfidence9", "poseConfidence10", "poseConfidence11", "poseConfidence12", "poseConfidence13", "poseConfidence14", "poseConfidence15", "poseConfidence16", "poseConfidence17"])          #create dataframe for pose confidence
            tempDataFrame = pd.concat([tempDataFrame, poseConfDataFrame], axis=1)          #concatenate keypoints and pose confidence dataframes
            poseDataFrame = pd.concat([poseDataFrame, tempDataFrame])          #append keypoints to dataframe
        poseDataFrame = poseDataFrame.iloc[1:]         #remove first row of dataframe
        poseDataFrame.reset_index(drop=True, inplace=True)          #reset index of dataframe
        return poseDataFrame
    
    def createPositionDataFrame(self, results):
        frame = 1
        for result in results:
            #print(result.boxes.xyxy.cpu().numpy())
            if frame == 1:
                if(result.boxes.xyxy.cpu().numpy().size == 0):
                    positionDataFrame = pd.DataFrame([(1,1,1,1,1)], columns=["px1", "py1", "px2", "py2", "positionConfidence"])
                    positionDataFrame.loc[len(positionDataFrame)] = None
                    positionDataFrame = positionDataFrame.iloc[1:]         #remove first row of dataframe
                    positionDataFrame.reset_index(drop=True, inplace=True)          #reset index of dataframe
                else:
                    positionDataFrame = pd.DataFrame(result.boxes.xyxy.cpu().numpy()[0].reshape(1,4), columns=["px1", "py1", "px2", "py2"])
                    positionDataFrame["positionConfidence"] = result.boxes.conf.cpu().numpy()[0].reshape(1,1)
                #positionDataFrame["class"] = result.boxes.cls.cpu().numpy()
            else:
                #print(result.boxes.xyxy.cpu().numpy())
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
        return positionDataFrame

    def createFullDataFrame(self, positionDataFrame, poseDataFrame):
        fullDataFrame = pd.concat([positionDataFrame, poseDataFrame], axis=1)          #concatenate position and pose dataframes
        fullDataFrame = fullDataFrame.reset_index(drop=True)
        return fullDataFrame
    
    def cleanUpFullDataFrame(self, fullDataFrame):
        fullDataFrame.drop(fullDataFrame[fullDataFrame['positionConfidence'] < 0.3].index, inplace=True)            #remove rows with confidence < 0.5
        fullDataFrame.drop(fullDataFrame[fullDataFrame['poseConfidence1'] < 0.3].index, inplace=True)            #remove rows with confidence < 0.5
        fullDataFrame.drop(fullDataFrame[fullDataFrame['poseConfidence2'] < 0.3].index, inplace=True)
        fullDataFrame.drop(fullDataFrame[fullDataFrame['poseConfidence3'] < 0.3].index, inplace=True)
        fullDataFrame.drop(fullDataFrame[fullDataFrame['poseConfidence4'] < 0.3].index, inplace=True)
        fullDataFrame.drop(fullDataFrame[fullDataFrame['poseConfidence5'] < 0.3].index, inplace=True)
        fullDataFrame.drop(fullDataFrame[fullDataFrame['poseConfidence6'] < 0.3].index, inplace=True)
        fullDataFrame.drop(fullDataFrame[fullDataFrame['poseConfidence7'] < 0.3].index, inplace=True)
        fullDataFrame.drop(fullDataFrame[fullDataFrame['poseConfidence8'] < 0.3].index, inplace=True)
        fullDataFrame.drop(fullDataFrame[fullDataFrame['poseConfidence9'] < 0.3].index, inplace=True)
        fullDataFrame.drop(fullDataFrame[fullDataFrame['poseConfidence10'] < 0.3].index, inplace=True)
        fullDataFrame.drop(fullDataFrame[fullDataFrame['poseConfidence11'] < 0.3].index, inplace=True)
        fullDataFrame.drop(fullDataFrame[fullDataFrame['poseConfidence12'] < 0.3].index, inplace=True)
        fullDataFrame.drop(fullDataFrame[fullDataFrame['poseConfidence13'] < 0.3].index, inplace=True)
        fullDataFrame.drop(fullDataFrame[fullDataFrame['poseConfidence14'] < 0.3].index, inplace=True)
        fullDataFrame.drop(fullDataFrame[fullDataFrame['poseConfidence15'] < 0.3].index, inplace=True)
        fullDataFrame.drop(fullDataFrame[fullDataFrame['poseConfidence16'] < 0.3].index, inplace=True)
        fullDataFrame.drop(fullDataFrame[fullDataFrame['poseConfidence17'] < 0.3].index, inplace=True)
        fullDataFrame.dropna(inplace=True, how='any')            #remove rows with NaN values
        fullDataFrame.drop(columns=["positionConfidence", "poseConfidence1", "poseConfidence2", "poseConfidence3", "poseConfidence4", "poseConfidence5", "poseConfidence6", "poseConfidence7", "poseConfidence8", "poseConfidence9", "poseConfidence10", "poseConfidence11", "poseConfidence12", "poseConfidence13", "poseConfidence14", "poseConfidence15", "poseConfidence16", "poseConfidence17"], inplace=True)            #remove confidence columns
        fullDataFrame = fullDataFrame.reset_index(drop=True)
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
