import cv2
from funk import roboflow_detect
from ultralytics import YOLO
from PIL import Image
#import pytesseract
import torch


def main():

    
    # Video fayli yo'lini belgilang
    video_path = 'road_rules_1.mp4'

    # Videoni yuklash
    cap = cv2.VideoCapture(video_path)

    # Videoni kadrlar bo'yicha o'qish
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        return_image = frame.copy()

        # model yolo
        model_yolo = YOLO("yolo11n.pt")
        results = model_yolo(frame, stream=True)

        for result in results:
            cv2.imshow("Result", result.plot())
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
