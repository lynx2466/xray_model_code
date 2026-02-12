from ultralytics import YOLO

# Load a model (use yolov8n.pt for fastest training)
model = YOLO("yolov8n.pt")

# Train
model.train(
    data="D:\\uhack_winner\\train\\chest-xray-yolo-4\\data.yaml",  # path to YAML
    epochs=2,
    imgsz=640,
    batch=8,
)
