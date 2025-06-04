from photoDetector import PhotoDetector
from videoDetector import VideoDetector
from detectionVisualizer import detectionVisualizer
from sys import exit
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from LSTMmodel import LSTMmodel
from LSTMtrain import LSTMtrain
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os



def main():  
    inputType = input("Enter input type (video/photo): ").strip().lower()
    if inputType == "video":

        whatJob = input("Enter what you want to do (train/predict/test): ").strip().lower()
        X = []
        y = []
        X = np.load(os.path.join("data", "X_0_1_new.npy"))         #X_0_1_backup.npy    #X_0_1_new.npy
        y = np.load(os.path.join("data", "y_0_1_new.npy"))         #y_0_1_backup.npy    #y_0_1_new.npy
        X, y = shuffle(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
        if whatJob == "train":
            
            # Get path to video
            #path_to_video = input("Enter path to video: ")                              #src/test/test_inputs/short.mp4         src/test/test_inputs/kickflip0.mov

            #extractFeatures(1, 207, X, y)     #trickID = 1 -> trickName = Kickflip          #114
            #extractFeatures(0, 204, X, y)     #trickID = 0 -> trickName = Ollie             #108

            #X = pad_sequences(X, dtype=float, padding='pre')
            #X = np.array(X)
            #y = np.array(y)
            #np.save(os.path.join("data", "X_0_1_new.npy"), X)
            #np.save(os.path.join("data", "y_0_1_new.npy"), y)
            print("X_train:\n", X_train)
            print("y_train:\n", y_train)
            print("X_test:\n", X_test)
            print("y_test:\n", y_test)
            model = LSTMmodel(X_train[0].shape, 1)
            trainer = LSTMtrain(model, 'lstm', batch_size=32, epochs=40000, validation_split=0.2)
            trainer.train(X_train, y_train)
            predictions = model.model.predict(X)
        elif whatJob == "test":
            model = LSTMmodel(X_train[0].shape, 1)
            trainer = LSTMtrain(model, 'lstm', batch_size=32, epochs=40000, validation_split=0.2)
            trainer.model.model = load_model("models/lstm.19221-0.476_82.hdf5.keras")             #dla ogólnych models/lstm.4587-0.473.hdf5.keras
            predictions = model.model.predict(X)
        else:
            model = LSTMmodel(X_train[0].shape, 1)
            trainer = LSTMtrain(model, 'lstm', batch_size=32, epochs=30000, validation_split=0.2)
            X = []
            path_to_video = input("Enter path to video: ")                 #src/test/test_inputs/ollieShort.mp4                        
            loadedModel = load_model("models/lstm.19221-0.476_82.hdf5.keras")

            detector = VideoDetector(model_path="models/yolo/COCO pretrained/yolo11n-pose.pt")
            detector.setClassesToTrack([0])                               # 0: person

            results = detector.detect(path_to_video)                     # detect with pose model
            poseDataFrame = detector.createPoseDataFrame(results)

            detector.setModel("models/yolo/COCO pretrained/yolo11n.pt")
            detector.setClassesToTrack([36])                               # 36: skateboard
            results = detector.detect(path_to_video)                                #detect again with normal model
            positionDataFrame = detector.createPositionDataFrame(results)

            fullDataFrame = detector.createFullDataFrame(positionDataFrame, poseDataFrame)
            fullDataFrame = detector.cleanUpFullDataFrame(fullDataFrame) 
            print(fullDataFrame)
            fullDataFrame = fullDataFrame.fillna(0)         

            scaler = MinMaxScaler(feature_range=(0,1))
            scaledFullDataFrame = scaler.fit_transform(fullDataFrame.astype(np.float32).values)          #scale data to 0-1
            X.append(scaledFullDataFrame)
            X = np.array(X)
            predictions = loadedModel.predict(X)
            print("PREDICTION: ", predictions)
            if predictions[0] < 0.5:
                print("Ollie")
            else:
                print("Kickflip")
            exit(0)

        # Print the results
        print("REAL: ", y)
        print("FIRST PREDICTIONS: ", predictions)
        for prediction in predictions:
            if prediction[0] < 0.5:
                prediction[0] = 0
            else:
                prediction[0] = 1
        print("PREDICTION:", predictions)
        print ("ACCURACY:", accuracy_score(y, predictions))
        print ("RECALL:", recall_score(y, predictions, average='weighted'))
        print ("PRECISION:", precision_score(y, predictions, average='weighted'))
        print ("F1:", f1_score(y, predictions, average='weighted'))
        confusionMatrix = confusion_matrix(y, predictions)
        sns.heatmap(confusionMatrix, annot=True, fmt='d')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

    else:
        path_to_image = input("Enter path to image: ")                              #src/test/test_inputs/skater.jpg

        detector = PhotoDetector(model_path="models/yolo/COCO pretrained/yolo11n-pose.pt")
        results = detector.detect(path_to_image)

        keypoints = results[0].keypoints.xy.cpu().numpy()
        keypointsconf = results[0].keypoints.conf.cpu().numpy()
        i = 1
        for keypoint in keypoints:                  #keypoints is a list of numpy arrays, each array is a list of keypoints
            tempDataFrame = pd.DataFrame(keypoint, columns=["x", "y"])
            tempDataFrame["confidence"] = keypointsconf[i-1]
            if i == 1:
                poseDataFrame = tempDataFrame
            else:
                poseDataFrame = pd.concat([poseDataFrame, tempDataFrame], axis=0)
            i += 1
        print(poseDataFrame)
        positionDataFrame = pd.DataFrame(results[0].boxes.xyxy.cpu().numpy(), columns=["x1", "y1", "x2", "y2"])
        positionDataFrame["confidence"] = results[0].boxes.conf.cpu().numpy()
        print(positionDataFrame)

        visualizer = detectionVisualizer(results=results)
        visualizer.visualizePhoto()
    
def extractFeatures(trickID, endRange, X, y):                       #trickName = Kickflip/Ollie
    if trickID == 0:
        trickName = "Ollie"
    elif trickID == 1:
        trickName = "Kickflip"
    for i in range(0, endRange):
        path_to_video = f"src/test/Tricks/{trickName}/{trickName}{i}.mov"

        detector = VideoDetector(model_path="models/yolo/COCO pretrained/yolo11n-pose.pt")
        detector.setClassesToTrack([0])                               # 0: person

        results = detector.detect(path_to_video)                    # detect with pose model
        poseDataFrame = detector.createPoseDataFrame(results)

        detector.setModel("models/yolo/COCO pretrained/yolo11n.pt")
        detector.setClassesToTrack([36])                               # 36: skateboard
        results = detector.detect(path_to_video)                                #detect again with normal model
        positionDataFrame = detector.createPositionDataFrame(results)

        fullDataFrame = detector.createFullDataFrame(positionDataFrame, poseDataFrame)
        fullDataFrame = detector.cleanUpFullDataFrame(fullDataFrame)
        print(fullDataFrame)

        fullDataFrame = fullDataFrame.fillna(0)         
        scaler = MinMaxScaler(feature_range=(0,1))
        scaledFullDataFrame = scaler.fit_transform(fullDataFrame.astype(np.float32).values)          #scale data to 0-1
        X.append(scaledFullDataFrame)          #append data to X
        y.append(trickID)          #append label to y
        

if __name__ == "__main__":
    main()
    exit(0)
