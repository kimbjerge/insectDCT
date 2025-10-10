import gc
import torch
torch.cuda.empty_cache()
gc.collect()
from ultralytics import YOLO

# Load a model
model = YOLO('/home/don/insectsDCT/runs/detect/insects5Motion/weights/best.pt')  # load a pretrained model (recommended for training)

# Train the model using color image datasets from "A deep learning pipeline .. floral environments" and "Motion informed object detection"
#results = model.val(data='insectsColor.yaml', batch=16, imgsz=1920, conf=0.25, device=0, show=False)
results = model.val(data='./data/insects5Motion.yaml', batch=16, imgsz=1920, conf=0.25, device=0, show=False, name='insects5MotionVal')
