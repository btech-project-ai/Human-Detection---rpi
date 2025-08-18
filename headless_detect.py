from ultralytics import YOLO
import cv2

def detect_image(image_path, model_path="best.pt", save_output=True, show=False):
    model = YOLO(model_path)
    results = model.predict(image_path, save=save_output, conf=0.5)

    for r in results:
        im_bgr = r.plot()
        
        if show:  # only show if GUI works
            cv2.imshow("Detections", im_bgr)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if save_output:
            output_path = "output_detected.jpg"
            cv2.imwrite(output_path, im_bgr)
            print(f"✅ Detection saved at {output_path}")

if __name__ == "__main__":
    detect_image("flood_image0413_1.png", show=False)
