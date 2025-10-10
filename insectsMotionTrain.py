import gc
import torch
torch.cuda.empty_cache()
gc.collect()
from ultralytics import YOLO

# Load a model
model = YOLO('yolo11m.pt')  # load a pretrained model (recommended for training)

# Train the model using MIE image datasets from "A deep learning pipeline .. floral environments" and "Motion informed object detection"

results = model.train(data='./data/insects5Motion.yaml', batch=4, epochs=70, imgsz=1920, flipud=0.5, device=0, show=False, name='insects5Motion')
#results = model.train(data='insectsMotion.yaml', batch=20, epochs=60, imgsz=1280, device=0, show=False)
