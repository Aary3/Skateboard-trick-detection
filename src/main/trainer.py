from ultraytics import YOLO
import torch

model = YOLO('C:/Users/piotr/yolov12-main/yolo11n-pose.pt')
print(f"Model is running on: {model.device}")

print("CUDA Available:", torch.cuda.is_available())  # Should print True
print("Current Device:", torch.cuda.current_device())  # Should print 0 (or another valid GPU index)
print("Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")

# Train the model
results = model.train(
  data='coco128.yaml',
  epochs=50, 
  batch=80, 
  imgsz=640,
  scale=0.5,  # S:0.9; M:0.9; L:0.9; X:0.9
  mosaic=1.0,
  mixup=0.0,  # S:0.05; M:0.15; L:0.15; X:0.2
  copy_paste=0.1,  # S:0.15; M:0.4; L:0.5; X:0.6
  workers=2,
  device="cuda:0",
)