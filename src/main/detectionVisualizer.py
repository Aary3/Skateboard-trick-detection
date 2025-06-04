import cv2

class detectionVisualizer:
    def __init__(self, results):
        self.results = results

    def visualizePhoto(self):
        img = self.results[0].plot()  # This plots the detections on the image

        # Display the image
        cv2.imshow('Detection Results', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def visualizeVideo(self, path_to_video):
        # Display the video with detections
        output_filename = path_to_video.replace('.mp4   ', '_output.avi')
        fps = 8  # Adjust based on input video
        frame_size = (640, 384)  # Modify based on video resolution

        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use 'MP4V' for .mp4
        out = cv2.VideoWriter(output_filename, fourcc, fps, frame_size)

        for result in self.results:
            frame = result.plot()  # Get the annotated frame
            frame = cv2.resize(frame, frame_size)  # Ensure consistent frame size
            out.write(frame)

        out.release()