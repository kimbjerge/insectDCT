import gc
import torch
torch.cuda.empty_cache()
gc.collect()
from ultralytics import YOLO

# Load a model
model = YOLO('/home/don/insectsDCT/runs/detect/insects5Color/weights/best.pt')  # load a pretrained model (recommended for training)

# Train the model using color image datasets from "A deep learning pipeline .. floral environments" and "Motion informed object detection"
#results = model.val(data='insectsColor.yaml', batch=16, imgsz=1920, conf=0.25, device=0, show=False)

file = open('detectTestColor.csv', 'w')

#header = "Dataset,TP,FP,FN,Precision,Recall,F1-score,mAP50,mAP50-95\n"
#header = "Dataset,Images,Insects,Precision,Recall,F1-score,mAP50,mAP50-95\n"
header = "Dataset,Precision,Recall,F1-score,mAP50,mAP50-95\n"
file.write(header)
testNames =  ['testGreenH', 'testHeather', 'testMBO', 'testOrchard', 'testPollW', 'testRTNI']
for testName in testNames:
    yamlFile = './data/' + testName + '.yaml'
    results = model.val(data=yamlFile, batch=16, imgsz=1920, conf=0.25, device=0, show=False, name=testName)
    #num_images = results.dataset.n
    #num_instances = sum(len(labels) for labels in results.dataset.labels)
    f1 = (2*results.box.mp*results.box.mr)/(results.box.mp+results.box.mr)
    #csvStr = f"{testName},{results.box.tp[0]},{results.box.fp[0]},{results.box.fn[0]},{results.box.mp:.4f},{results.box.mr:.4f},{f1:.4f},{results.box.map50:.4f},{results.box.map:.4f}\n"
    #csvStr = f"{testName},{num_images},{num_labels},{results.box.mp:.4f},{results.box.mr:.4f},{f1:.4f},{results.box.map50:.4f},{results.box.map:.4f}\n"
    csvStr = f"{testName},{results.box.mp:.4f},{results.box.mr:.4f},{f1:.4f},{results.box.map50:.4f},{results.box.map:.4f}\n"
    print(csvStr)
    file.write(csvStr)
file.close()

