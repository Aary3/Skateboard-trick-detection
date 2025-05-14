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
    #a1 = [f"kx{i//2+1}" if i%2==0 else f"ky{i//2+1}" for i in range(34)]
    #a2 =[f"poseConfidence{i+1}" for i in range(17)]
    #print(a1+a2)
    #modelType = input("Enter model type (normal/pose/full): ").strip().lower()
    # Load a pretrained YOLO11n model
    #if(modelType == "pose"):
    #    pathToModel = "models/yolo/COCO pretrained/yolo11n-pose.pt"
    #else:
    #    pathToModel = "models/yolo/COCO pretrained/yolo11n.pt"

    #get input and visualize
    
    inputType = input("Enter input type (video/photo): ").strip().lower()
    if inputType == "video":
        isTraining = input("Is this training? (y/n): ").strip().lower()
        if isTraining == "y":
            X = []
            y = []
            # Get path to video
            #path_to_video = input("Enter path to video: ")                              #src/test/test_inputs/short.mp4         src/test/test_inputs/kickflip0.mov

            #extractFeatures(1, 207, X, y)     #trickID = 1 -> trickName = Kickflip          #114
            #extractFeatures(0, 204, X, y)     #trickID = 0 -> trickName = Ollie             #108

            #X = pad_sequences(X, dtype=float, padding='pre')
            #X = np.array(X)
            #y = np.array(y)
            #np.save(os.path.join("data", "X_0_1_new.npy"), X)
            #np.save(os.path.join("data", "y_0_1_new.npy"), y)

            X = np.load(os.path.join("data", "X_0_1_new.npy"))         #X_0_1_backup.npy
            y = np.load(os.path.join("data", "y_0_1_new.npy"))         #y_0_1_backup.npy

            #shuffling the data
            X, y = shuffle(X, y)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
            #X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]*X_train.shape[2], 1))
            #X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]*X_test.shape[2], 1))
            print("X_train:\n", X_train)
            print("y_train:\n", y_train)
            print("X_test:\n", X_test)
            print("y_test:\n", y_test)


            #training
            model = LSTMmodel(X_train[0].shape, 1)
            trainer = LSTMtrain(model, 'lstm', batch_size=32, epochs=10000, validation_split=0.2)
            #trainer.model.model = load_model("models/lstm.1323-0.498.hdf5.keras")
            trainer.train(X_train, y_train)

            #predictions = model.model.predict(X_test)
            predictions = model.model.predict(X)
        else:
            X = []
            path_to_video = input("Enter path to video: ")                              
            loadedModel = load_model("models/lstm.1323-0.498.hdf5.keras")
            # Create a VideoDetector instance
            detector = VideoDetector(model_path="models/yolo/COCO pretrained/yolo11n-pose.pt")
            detector.setClassesToTrack([0])                               # 0: person
            # Perform detection on the video
            results = detector.detect(path_to_video)
            poseDataFrame = detector.createPoseDataFrame(results)          #create pose dataframe

            detector.setModel("models/yolo/COCO pretrained/yolo11n.pt")
            detector.setClassesToTrack([36])                               # 36: skateboard
            results = detector.detect(path_to_video)                                #detect again with normal model
            positionDataFrame = detector.createPositionDataFrame(results)

            fullDataFrame = detector.createFullDataFrame(positionDataFrame, poseDataFrame)          #create full dataframe
            fullDataFrame = detector.cleanUpFullDataFrame(fullDataFrame)          #clean up full dataframe
            #finalDataFrame = detector.createFinalDataFrame(fullDataFrame)          #create final dataframe
            print(fullDataFrame)
            #fullDataFrame = fullDataFrame.dropna(how = 'any')          #drop rows with NaN values
            fullDataFrame = fullDataFrame.fillna(0)          #fill NaN values with 0
            #if(fullDataFrame.shape[0] < 30):          #if dataframe has less then 30 rows, skip
            #    print("Not enough quality data, skipping...")
            #    continue
            scaler = MinMaxScaler(feature_range=(0,1))
            scaledFullDataFrame = scaler.fit_transform(fullDataFrame.astype(np.float32).values)          #scale data to 0-1
            X.append(scaledFullDataFrame)
            X = np.array(X)
            predictions = loadedModel.predict(X)
            if predictions[0] < 0.5:
                print("Ollie")
            else:
                print("Kickflip")
            

        #print("REAL:", y_test)
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

        # Create a detectionVisualizer instance and visualize the video
        #visualizer = detectionVisualizer(results=results)
        #visualizer.visualizeVideo(path_to_video)
    else:
        # Get path to image
        path_to_image = input("Enter path to image: ")                              #src/test/test_inputs/skater.jpg
        # Create a PhotoDetector instance
        detector = PhotoDetector(model_path="models/yolo/COCO pretrained/yolo11n-pose.pt")
        # Perform detection on the image
        results = detector.detect(path_to_image)

        # Print the results
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
        #positionDataFrame["class"] = results[0].boxes.cls.cpu().numpy()
        print(positionDataFrame)

        # Create a detectionVisualizer instance and visualize the imagegit basic commands
        visualizer = detectionVisualizer(results=results)
        visualizer.visualizePhoto()
    
def extractFeatures(trickID, endRange, X, y):                       #trickName = Kickflip/Ollie
    if trickID == 0:
        trickName = "Ollie"
    elif trickID == 1:
        trickName = "Kickflip"
    for i in range(0, endRange):
        path_to_video = f"src/test/Tricks/{trickName}/{trickName}{i}.mov"
        # Create a VideoDetector instance
        detector = VideoDetector(model_path="models/yolo/COCO pretrained/yolo11n-pose.pt")
        detector.setClassesToTrack([0])                               # 0: person
        # Perform detection on the video
        results = detector.detect(path_to_video)
        poseDataFrame = detector.createPoseDataFrame(results)          #create pose dataframe

        detector.setModel("models/yolo/COCO pretrained/yolo11n.pt")
        detector.setClassesToTrack([36])                               # 36: skateboard
        results = detector.detect(path_to_video)                                #detect again with normal model
        positionDataFrame = detector.createPositionDataFrame(results)

        fullDataFrame = detector.createFullDataFrame(positionDataFrame, poseDataFrame)          #create full dataframe
        fullDataFrame = detector.cleanUpFullDataFrame(fullDataFrame)          #clean up full dataframe
        #finalDataFrame = detector.createFinalDataFrame(fullDataFrame)          #create final dataframe
        print(fullDataFrame)
        #fullDataFrame = fullDataFrame.dropna(how = 'any')          #drop rows with NaN values
        fullDataFrame = fullDataFrame.fillna(0)          #fill NaN values with 0
        #if(fullDataFrame.shape[0] < 30):          #if dataframe has less then 30 rows, skip
        #    print("Not enough quality data, skipping...")
        #    continue
        scaler = MinMaxScaler(feature_range=(0,1))
        scaledFullDataFrame = scaler.fit_transform(fullDataFrame.astype(np.float32).values)          #scale data to 0-1
        X.append(scaledFullDataFrame)          #append data to X
        y.append(trickID)          #append label to y
        

if __name__ == "__main__":
    main()
    exit(0)
