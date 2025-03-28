from ultralytics import YOLO
import cv2
from sys import exit

def main():
    # Load a pretrained YOLO11n model
    model = YOLO("models/yolo/COCO pretrained/yolo11n-pose.pt")

    #Get path to image
    path_to_image = input("Enter path to image: ")

    #/home/piotr-sosnowski/Downloads/skater2.jpg
    # Perform object detection on an image
    results = model(path_to_image)  # Predict on an image
    img = results[0].plot()  # This plots the detections on the image

    # Display the image
    cv2.imshow('Detection Results', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    exit(0)