from ultralytics import YOLO

# Settings
model_name = "yolov8n.pt"  # tiny model
data_path = "yolo_dataset/dataset.yaml"

epochs = 100

# Load model
model = YOLO(model_name)

# Start fresh training
results = model.train(
    data=data_path,
    epochs=epochs,
    imgsz=640,
    name="road_signs_v1"
)

