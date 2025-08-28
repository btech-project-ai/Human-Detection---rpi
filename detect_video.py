from ultralytics import YOLO
import cv2

def detect_video(video_path, model_path=r"best.onnx", save_output=True):
    """
    Detect objects in a video using a trained YOLO model.
    
    Args:
        video_path (str): Path to the input video.
        model_path (str): Path to the trained YOLO model (.pt file).
        save_output (bool): If True, saves the output video with detections.
    """
    # Load YOLO model
    model = YOLO(model_path)

    # Open video
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define video writer if saving output
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec
        out = cv2.VideoWriter("output_detected.mp4", fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection on each frame
        results = model.predict(frame, conf=0.5, verbose=False)

        for r in results:
            im_bgr = r.plot()  # draw bounding boxes

            # Show detections in a window
            cv2.imshow("Human Detection", im_bgr)

            # Save output video
            if save_output:
                out.write(im_bgr)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if save_output:
        out.release()
        print("✅ Detection video saved at output_detected.mp4")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Example usage
    video_path = r"test_video.mp4"  # replace with your test video
    detect_video(video_path)
