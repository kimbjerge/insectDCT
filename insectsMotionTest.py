import gc
import torch
torch.cuda.empty_cache()
gc.collect()
from ultralytics import YOLO

# Load a model
model = YOLO('/home/don/insectsDCT/runs/detect/insects5Motion/weights/best.pt')  # load a pretrained model (recommended for training)

# Train the model using color image datasets from "A deep learning pipeline .. floral environments" and "Motion informed object detection"
#results = model.val(data='insectsColor.yaml', batch=16, imgsz=1920, conf=0.25, device=0, show=False)

file = open('detectTestMotion.csv', 'w')

header = "Dataset,Precision,Recall,F1-score,mAP50,mAP50-95\n"
file.write(header)
testNames = ['testGreenHm', 'testHeatherm', 'testMBOm', 'testOrchardm', 'testPollWm', 'testRTNIm']
for testName in testNames:
    yamlFile = './data/' + testName + '.yaml'
    results = model.val(data=yamlFile, batch=16, imgsz=1920, conf=0.25, device=0, show=False, name=testName)
    f1 = (2*results.box.mp*results.box.mr)/(results.box.mp+results.box.mr)
    csvStr = f"{testName},{results.box.mp:.4f},{results.box.mr:.4f},{f1:.4f},{results.box.map50:.4f},{results.box.map:.4f}\n"
    print(csvStr)
    file.write(csvStr)
file.close()

