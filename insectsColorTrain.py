import gc
import torch
torch.cuda.empty_cache()
gc.collect()
from ultralytics import YOLO

# Load a model
model = YOLO('yolo11m.pt')  # load a pretrained model (recommended for training)

# Train the model using color image datasets from "A deep learning pipeline .. floral environments" and "Motion informed object detection"
#results = model.train(data='insectsColor.yaml', batch=4, epochs=60, imgsz=1920, flipud=0.5, device=[0,1], show=False)
results = model.train(data='./data/insects5Color.yaml', batch=4, epochs=70, imgsz=1920, flipud=0.5, device=0, show=False, name='insects5Color')
