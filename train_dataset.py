from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # Load a pretrained model
results = model.train(data="dataset_license_plate/data.yaml", epochs=5, imgsz=640)