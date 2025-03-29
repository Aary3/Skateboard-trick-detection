from photoDetector import PhotoDetector
from videoDetector import VideoDetector
from detectionVisualizer import detectionVisualizer
from sys import exit

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

        # Create a detectionVisualizer instance and visualize the image
        visualizer = detectionVisualizer(results=results)
        visualizer.visualizePhoto()
    

if __name__ == "__main__":
    main()
    exit(0)