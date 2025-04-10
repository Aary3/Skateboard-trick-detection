from photoDetector import PhotoDetector
from videoDetector import VideoDetector
from detectionVisualizer import detectionVisualizer
from sys import exit
import pandas as pd
import torch


def main():
    modelType = input("Enter model type (normal/pose/full): ").strip().lower()
    # Load a pretrained YOLO11n model
    if(modelType == "pose"):
        pathToModel = "models/yolo/COCO pretrained/yolo11n-pose.pt"
    else:
        pathToModel = "models/yolo/COCO pretrained/yolo11n.pt"

    #get input and visualize
    inputType = input("Enter input type (video/photo): ").strip().lower()
    if inputType == "video":
        # Get path to video
        path_to_video = input("Enter path to video: ")                              #src\test\test_inputs\eltoro.mp4
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
        

        # Create a detectionVisualizer instance and visualize the video
        visualizer = detectionVisualizer(results=results)
        visualizer.visualizeVideo(path_to_video)
    else:
        # Get path to image
        path_to_image = input("Enter path to image: ")                              #src\test\test_inputs\skater.jpg
        # Create a PhotoDetector instance
        detector = PhotoDetector(model_path=pathToModel)
        # Perform detection on the image
        results = detector.detect(path_to_image)

        # Print the results
        if modelType == "pose":
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
        else:
            positionDataFrame = pd.DataFrame(results[0].boxes.xyxy.cpu().numpy(), columns=["x1", "y1", "x2", "y2"])
            positionDataFrame["confidence"] = results[0].boxes.conf.cpu().numpy()
            #positionDataFrame["class"] = results[0].boxes.cls.cpu().numpy()
            print(positionDataFrame)

        # Create a detectionVisualizer instance and visualize the image
        visualizer = detectionVisualizer(results=results)
        visualizer.visualizePhoto()
    

if __name__ == "__main__":
    main()
    exit(0)