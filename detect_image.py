from ultralytics import YOLO
import cv2

def detect_image(image_path, model_path=r"best_v2.pt", save_output=True):
    """
    Detect objects in a single image using a trained YOLO model.
    
    Args:
        image_path (str): Path to the input image.
        model_path (str): Path to the trained YOLO model (.pt file).
        save_output (bool): If True, saves the output image with detections.
    """
    # Load model
    model = YOLO(model_path)

    # Run detection
    results = model.predict(image_path, save=save_output, conf=0.5)

    # Show detection result
    for r in results:
        im_bgr = r.plot()  # plot detection boxes
        cv2.imshow("Detections", im_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Optionally save output
        if save_output:
            output_path = "output_detected.jpg"
            cv2.imwrite(output_path, im_bgr)
            print(f"✅ Detection saved at {output_path}")


if __name__ == "__main__":
    # Example usage
    image_path = r"human.jpg"  # put your test image here
    detect_image(image_path)
