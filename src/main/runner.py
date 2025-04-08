from photoDetector import PhotoDetector
from videoDetector import VideoDetector
from detectionVisualizer import detectionVisualizer
from sys import exit
import pandas as pd


def main():
    modelType = input("Enter model type (normal/pose): ").strip().lower()
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
        detector = VideoDetector(model_path=pathToModel)
        # Perform detection on the video
        results = detector.detect(path_to_video)

        # Print the results
        if modelType == "pose":
            oldDataFrame = None
            for result in results:
                if ((result.keypoints.xy is None) or (result.keypoints.conf is None)):
                    continue                                #skip if no keypoints are detected
                keypoints = result.keypoints.xy.cpu().numpy()
                keypointsconf = result.keypoints.conf.cpu().numpy()
                i = 1
                for keypoint in keypoints:                  #keypoints is a list of numpy arrays, each array is a list of keypoints
                    tempDataFrame = pd.DataFrame(keypoint, columns=["x", "y"])
                    tempDataFrame["confidence"] = keypointsconf[i-1]
                    if i == 1:
                        poseDataFrame = tempDataFrame
                    else:
                        poseDataFrame = pd.concat([poseDataFrame, tempDataFrame], axis=0)
                    i += 1
                if oldDataFrame is None:
                    oldDataFrame = poseDataFrame.copy()
                else:
                    oldDataFrame = pd.concat([oldDataFrame, poseDataFrame], axis=0)
            poseDataFrame = oldDataFrame.copy()
            poseDataFrame = poseDataFrame.reset_index(drop=True)
            print(poseDataFrame)
        else:
            frame = 1
            for result in results:
                if frame == 1:
                    positionDataFrame = pd.DataFrame(result.boxes.xyxy.cpu().numpy(), columns=["x1", "y1", "x2", "y2"])
                    positionDataFrame["confidence"] = result.boxes.conf.cpu().numpy()
                    positionDataFrame["class"] = result.boxes.cls.cpu().numpy()
                else:
                    tempDataFrame = pd.DataFrame(result.boxes.xyxy.cpu().numpy(), columns=["x1", "y1", "x2", "y2"])
                    tempDataFrame["confidence"] = result.boxes.conf.cpu().numpy()
                    tempDataFrame["class"] = result.boxes.cls.cpu().numpy()
                    positionDataFrame = pd.concat([positionDataFrame, tempDataFrame], axis=0)
                frame += 1
            print(positionDataFrame)    

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
            positionDataFrame["class"] = results[0].boxes.cls.cpu().numpy()
            print(positionDataFrame)

        # Create a detectionVisualizer instance and visualize the image
        visualizer = detectionVisualizer(results=results)
        visualizer.visualizePhoto()
    

if __name__ == "__main__":
    main()
    exit(0)