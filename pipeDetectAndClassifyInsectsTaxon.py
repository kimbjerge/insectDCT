# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:11:38 2025

@author: Kim Bjerge
"""

import os
import gc
import cv2
import csv
import torch
import argparse
import datetime 
import pickle

#from common.cnn_classifier import CnnClassifier # Uncomment if flat classifier should be used
from common.hierarchical_classifier import HierarchicalClassifier
from common.motionEnhancement import MotionEnhancement

torch.cuda.empty_cache()
gc.collect()
from ultralytics import YOLO

from PIL import Image
from PIL.ExifTags import TAGS

results_dir = './detections/'
crops_dic_insect = './crops/'

useMambo = False

# MAMBO
if useMambo:    
    imgWidth = 3840 # Grade Pro 8MP
    imgHeight = 2160
    #imgWidth = 4224
    #imgHeight = 2376
else:
# Jordan and Simon
    imgWidth = 1920
    imgHeight = 1080
    
labelNames = ['Insect'] # YOLO Only one label

# labelSpeciesNames = ["A1-Coccinellidae", "B2-Coleoptera", "C3-Background", "D4-Bombus", "E5-Syrphidae", 
#                      "F6-Lepidoptera", "G7-Araneae", "H8-Formidicidae", "I9-Diptera", "J10-Hemiptera", 
#                      "K11-Isopoda", "L12-Unspecified", "N13-Hymenoptera", "O14-Orthoptera", "P15-Rhagnoycha_fulva", 
#                      "Q16-Satyrinae", "R17-Aglais_urticea", "S18-Odonata", "T19-Apis_mellifera"]

# Specises for flat CnnClassifier with 19 classes
labelSpeciesNames = ["Coccinellidae", "Coleoptera", "Background", "Bombus", "Syrphidae", 
                     "Lepidoptera", "Araneae", "Formidicidae", "Diptera", "Hemiptera", 
                     "Isopoda", "Unspecified", "Hymenoptera", "Orthoptera", "Rhagnoycha fulva", 
                     "Satyrinae", "Aglais urticea", "Odonata", "Apis mellifera"]

def createHierarchicalClassifier(weights_file, label_file, threshold_file, img_size=128, stdThreshold=2.0, device='cuda:0'):
    
    with open(label_file, 'rb') as f:
        _, hierarchyL1, hierarchyL2, labelsL1, labelsL2, labelsL3, _, _, _, _, _, _ = pickle.load(f)
        print("Labels and hierarchy dependency loaded from ", label_file)
        print("=============================================================================================")
        print("L1 classes", labelsL1, len(labelsL1))
        print("=============================================================================================")
        print("L2 classes", labelsL2, len(labelsL2))
        print("=============================================================================================")
        print("L3 classes", labelsL3, len(labelsL3))
        print("=============================================================================================")
        print("L2 -> L1 dependency", hierarchyL1)
        print("=============================================================================================")
        print("L3 -> L2 dependency", hierarchyL2)
        print("=============================================================================================")
    
    classifier = HierarchicalClassifier(hierarchyL1, hierarchyL2, labelsL1, labelsL2, labelsL3, img_size=img_size, stdThreshold=stdThreshold, device=device)
    classifier.loadmodel(weights_file, threshold_file)
    
    return classifier
    
#%% Function to classify insects in 19 groups of taxa
def classifyInsect(classifier, image, xc, yc, w, h, cropName, border=1, createCrops=False): # Border=10 for crops

    height, width, channels = image.shape
    
    w = (w + border*2)
    h = (h + border*2)
    if h > w: 
        wh2 = int(round(h/2))
    else:
        wh2 = int(round(w/2))
    
    x1 = xc-wh2
    if x1 < 0:
       x1 = 0
    x2 = xc+wh2
    if x2 >= width:
        x2 = width-1
    y1 = yc-wh2
    if y1 < 0: 
        y1 = 0
    y2 = yc+wh2
    if y2 >= height:
        y2 = height-1
        
    imgCrop = image[y1:y2, x1:x2,  :].copy()
    
    if type(classifier) is HierarchicalClassifier:    
        line, level, index, species, probability = classifier.makePrediction(imgCrop)
    else: # Flat CnnClassifier
        level = 0
        index, species, probability = classifier.makePrediction(imgCrop)
        line = species + ',' + str(level) + ',' + str(probability) 
    
    print(line)         
    
    if createCrops:
        dirName = str(level) + '-' + species
        if os.path.exists(crops_dic_insect + dirName) == False:
            print("Create directory:", crops_dic_insect + dirName)
            os.mkdir(crops_dic_insect + dirName)
        imgNameCrop = cropName + '_' + str(xc) + '_' + str(yc) + '_' + str(w) + '_' + str(h) + '.jpg'
        cv2.imwrite(crops_dic_insect + dirName + '/' + imgNameCrop, imgCrop)
    
    return line, level, index, species, probability

#%% Return datetime based on image filename with the format: camera_YYYY_MM_DD_HH_MM_SS.jpg
def getFrameTime(filePath, image_filename, useTimeExif): 

    if useTimeExif:
        # open the image
        image = Image.open(filePath+image_filename)
         
        # extracting the exif metadata
        exifdata = image.getexif()
         
        # looping through all the tags present in exifdata
        for tagid in exifdata:
            # getting the tag name instead of tag id
            tagname = TAGS.get(tagid, tagid)
            if tagname == "DateTime": # Check tag date time
                # passing the tagid to get its respective value
                value = exifdata.get(tagid)
                # printing the final result
                #print(f"{tagname:25}: {value}")
                
                # reformat time stamp to "YYYYMMDDHHMMSS"
                timestamp = value.replace(':', '')
                dateTimeStr = timestamp.replace(' ', '')
           
        # close the image
        image.close()
    
    else:    
        nameSplit = image_filename.split('.')[0].split('_')
        dateTimeStr = nameSplit[1] + nameSplit[2] + nameSplit[3] + nameSplit[4] + nameSplit[5] + nameSplit[6] # Format: YYYYMMDDHHMMSS
    
    image_time = datetime.datetime.strptime(dateTimeStr, "%Y%m%d%H%M%S")

    return image_time, dateTimeStr

#%% Pipe line to process each frame using motion informed enhancement if useMotion=True
def processFrame(frame, frame_time, frame_count, frames_after, useMotion, saveMovie, args, filename='', prevFilename=''):
    # Global variables:  MIE, modelDetector, modelClassifier, movie_writer, csv_writer
       
    # Use timestamp from current frame
    timestamp_year_str = frame_time.strftime("%Y")
    timestamp_date_str = frame_time.strftime("%Y%m%d")
    timestamp_time_str = frame_time.strftime("%H%M%S")

    if useMotion:
        frame, imgPrev = MIE.motion_image(frame)

    # Run YOLO inference on the frame
    results = modelDetector.predict(frame, batch=1, conf=args.confidence, device=args.device) # Automatic scales to HD image size
    #results = modelDetector.predict(frame, imgsz=imgWidth, batch=1, conf=args.confidence, device=args.device)
    
    if useMotion and args.videoMIE == False:
        frame = imgPrev
   
    # View results
    insectFound = False
    insectsFound = 0
    for r in results:
        boxes = r.boxes.cpu()
        boxes = boxes.numpy()
        for box in boxes:
            xyxy = box.xyxy # top-left-x, top-left-y, bottom-right-x, bottom-right-y
            # Wrong xywh = boxes.xywh  # center-x, center-y, width, height KBE????
            xywh = box.xywh  # center-x, center-y, width, height
            if xyxy.size > 0: # Save object found
                #print(xyxy, box.cls, box.conf)
                clas = int(box.cls[0])
                conf = int(round(box.conf[0]*100))
                x1 = int(round(xyxy[0][0]))
                y1 = int(round(xyxy[0][1]))
                x2 = int(round(xyxy[0][2]))
                y2 = int(round(xyxy[0][3]))
                
                if type(modelClassifier) is not int:
                    line,  level, speciesIdx, speciesName, probability = classifyInsect(modelClassifier, frame,
                                                                                        int(round(xywh[0][0])), 
                                                                                        int(round(xywh[0][1])), 
                                                                                        int(round(xywh[0][2])), 
                                                                                        int(round(xywh[0][3])),
                                                                                        str(frame_count))
                    prob = round(probability*10000)/100 # percentage with two decimals
                else:
                    line = ''
                    level = 0
                    speciesIdx = -1
                    speciesName = "Unidentified"
                    prob = 0

                taxaSure = not (speciesIdx < 0) # Below threshold or wrong hierarchy
                
                if useMotion and prevFilename != '':
                    saveFilename = prevFilename
                else:
                    saveFilename = filename

                if args.CSVformat == "tracking": # Format used for tracing insects
                    #headerLine = "system,trap,date,time,detectConf,detectId,x1,y1,x2,y2,fileName\n" (Old format)
                    #headerLine = "trap,trapId,date,time,taxaConf,taxaLabel,taxaId,taxaLevel,frameId,x1,y1,x2,y2,fileName\n" 
                    input_variable = [args.camera, int(args.camera[2]), timestamp_date_str, timestamp_time_str, prob, speciesName, speciesIdx+1, level, frame_count, x1, y1, x2, y2, saveFilename]
                else: # Format used for tracking moths
                    #headerLine = "year,trap,date,time,detectConf,detectId,x1,y1,x2,y2,fileName,taxaLabel,taxaId,taxaLevel,taxaConf,taxaSure,frameId\n"
                    input_variable = [timestamp_year_str, args.camera, timestamp_date_str, timestamp_time_str, 
                                      conf, clas, x1, y1, x2, y2, saveFilename, speciesName, speciesIdx+1, level, prob, taxaSure, frame_count]
                
                print(input_variable)
                csv_writer.writerow(input_variable)
                csvfile.flush()
                
                if type(modelClassifier) is HierarchicalClassifier:
                    #headerLine = "Label1,LabelId1,Conf1,Above1,Label2,LabelId2,Conf2,Above2,Label3,LabelId3,Conf3,Above3,Checked,frameId\n"
                    line += ',' + str(frame_count) + '\n'
                    csvfileInfo.write(line)
                    csvfileInfo.flush()
                    
                if saveMovie:
                    insectFound = True
                    frames_after = store_frames_after
                    color = (0,0,255) # Red
                    if insectsFound % 3 == 1:
                        color = (0,255,0) # Green
                    if insectsFound % 3 == 2:
                        color = (255,0,0) # Blue
                    insectsFound += 1
                    
                    if type(modelClassifier) is int:
                        insectName = labelNames[clas-1] + ' (' + str(conf)+ ')'
                    else: # Species classifier used
                        if taxaSure:
                            if prob < 10.0:
                                probDisp = round(prob*10)/10 # display probability (%) with one decimal
                            else:
                                probDisp = int(round(prob)) # round to integer if more than 10%
                            insectName = speciesName + ' (' + str(level) + '-' + str(probDisp) + ')'
                        else:
                            insectName = speciesName
                    y = int(round(y1-20))
                    
                    if useMambo:                                    
                        cv2.rectangle(frame,(x1,y1-10),(x2,y2), color, 8) # 4 HD
                        cv2.putText(frame, insectName, (x1,y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 6, cv2.LINE_AA) # 1, 2 HD
                    else:
                        cv2.putText(frame, insectName, (x1,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA) # 1, 2 HD                
                        cv2.rectangle(frame,(x1,y1-10),(x2,y2), color, 4) # 4 HD
    
    dateTimeStr =  timestamp_date_str + ' ' + timestamp_time_str
    
    if useMambo:
        cv2.putText(frame, dateTimeStr, (40,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 6, cv2.LINE_AA) # (20,40) 1, 2 HD
    else:
        cv2.putText(frame, dateTimeStr, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA) # (20,40) 1, 2 HD
              
    if insectFound or frames_after > 0:
        frames_after -= 1
        image = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        movie_writer.write(image) # cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    prevFilename = filename
    return prevFilename, frames_after 

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--yoloWeights', default='./runs/detect/insects3Motion/weights/best.pt') #Directory that contains motion models
    #parser.add_argument('--yoloWeights', default='./runs/detect/insects3Color/weights/best.pt') #Directory that contains color models
    parser.add_argument('--optimized', default='') # Optimized for embedded processing (ncnn)

    #parser.add_argument('--classifier', default='./models_save/EfficientNetB4-19cls-50-Ext-Finetuned.keras') # 224x224 F1 0.85
    parser.add_argument('--classifier', default='')
    
    # First model trained on dataset from NI2 and NI1 (Nature Impact - Wingscapes cameras and Logitech webcameras)
    #parser.add_argument('--hierachical', default='./models_save/HierarchicalClassifier_13052025.pth') # 128x128 F1: L1 0.93, L2 0.76, L3 0.68
    #parser.add_argument('--labels', default='./models_save/HierarchicalLabels3L_13052025.pkl')
    #parser.add_argument('--thresholds', default='./models_save/HierarchicalThresholds_13052025_TH2.csv') # Use thresholds below = mean-2*std
    #parser.add_argument('--thresholds', default='./models_save/HierarchicalThresholds_13052025_TH3.csv') # Use thresholds below = mean-3*std
    
    # Model trained with added dataset "sorted_orchard_crops" from UFZ (+Camera pi3 camera images)
    parser.add_argument('--hierachical', default='./models_save/HierarchicalClassifierV2_30082025.pth') # 128x128 F1: L1 0.93, L2 0.76, L3 0.68
    parser.add_argument('--labels', default='./models_save/HierarchicalLabels3LV2_30082025.pkl')
    parser.add_argument('--thresholds', default='./models_save/HierarchicalThresholdsV2_30082025_TH3.csv') # Use thresholds below = mean-3*std
    
    parser.add_argument('--thresholdStd', default='0.0', type=float) # Use threshold below = mean - thresholdStd*std (Default 0.0 uses thresholds csv file)
    parser.add_argument('--project', default='UFZ') # Default UFZ else use MAMBO used for naming CSV files
    
    parser.add_argument('--useExifTime', default='', type=bool) # Default (False) use date time in filename or from exif file data (True)
    parser.add_argument('--video', default='')
    #parser.add_argument('--video', default='/home/don/yolov5r/yolov5/PollNI2/pi12024_05_24_05_00_01.mp4')
    #parser.add_argument('--video', default='/home/don/yolov5r/yolov5/PollNI2/pi102024_06_11_05_00_02.mp4')
    #parser.add_argument('--video', default='./datasets/pi22025_04_06_23_11_00.mov') # 30 fps -> 10 fps (stride 3) Jordan
    #parser.add_argument('--video', default='./datasets/beemoni12025_04_03_12_59_47.mp4') # 45 fps -> 5 fps (stride 5) Jordanargs.device
    parser.add_argument('--images', default='./images/pi1_2025_02_21/')
    #parser.add_argument('--confidence', default='0.374', type=float) # insect3Color best F1-score 0.93
    parser.add_argument('--confidence', default='0.401', type=float) # insect3Motion best F1-score 0.93
    parser.add_argument('--device', default='cuda:0') # used for GPU or CPU processing (cuda:X or cpu) 
    parser.add_argument('--camera', default='pi1') # Overwritten by camera specified in image filename for time-lapse images
    parser.add_argument('--frame_stride', default='2', type=int) # for video, not used for images
    parser.add_argument('--scale', default='0.45', type=float) # Scale factor used for creating result video
    parser.add_argument('--videoMIE', default='', type=bool) # Show video with Motion Informed Enhanced frames (True)
    parser.add_argument('--moviePredict', default='movie') # Save movie with bounding boxes and classifications (Empty string no movie saved)
    parser.add_argument('--CSVformat', default='tracking') # Store result file in format used by insectTracking
    
    args = parser.parse_args() 
    print(args)
    
    frame_stride = args.frame_stride # Video recorded with 1 fps
    fps=1/frame_stride
    store_frames_after = 1 # Video 2

    prevFilename = ''
    useMotion = False
    if "Motion" in args.yoloWeights: # Use motion informed enhancement in detecting insects
        useMotion = True
        MIE = MotionEnhancement()
    
    # Load the YOLO11 insect detector model
    if args.optimized == "ncnn":
        yoloNCNN = args.yoloWeights.replace(".pt", "_ncnn_model")
        if not os.path.exists(yoloNCNN):
            modelDetector = YOLO(args.yoloWeights)
            modelDetector.export(format="ncnn", half=False) # export model to optimized NCNN format (FP16)
        modelDetector = YOLO(yoloNCNN) # load optimized NCNN model   
    else:
        modelDetector = YOLO(args.yoloWeights)  # load trained model
    
    # Load the insect classifier model
    modelClassifier = 0
    if args.classifier != '': # Classify detected insects into categories of taxa
        print("Loading insect classifier model", args.classifier)
        print(labelSpeciesNames)
        #modelClassifier = CnnClassifier(args.classifier, labelSpeciesNames, (224,224)) # Uncomment if flat classifier should be used
        
    if args.hierachical != '':
        print("Loading hierarchical insect classifier model", args.hierachical)
        modelClassifier = createHierarchicalClassifier(args.hierachical, args.labels, args.thresholds, 128, stdThreshold=args.thresholdStd, device=args.device)

    # Open the input video file if specified
    video_path = args.video
    if video_path != '': # Process video file
        cap = cv2.VideoCapture(video_path)
        videoSplit = args.video.split('_')
        dateTimeStr = "2025" + videoSplit[1] + videoSplit[2] + videoSplit[3] + videoSplit[4] + "00" # Format: YYYYMMDDHHMMSS
        start_time = datetime.datetime.strptime(dateTimeStr, "%Y%m%d%H%M%S")
        #csvFilename = results_dir + args.video.split('/')[-1].replace('mp4','csv')
        imagesSubDir = args.video.split('/')[-1]
        imagesSubDir = imagesSubDir.split('.')[0]
        csvFilename = results_dir + imagesSubDir + '-CL.csv' # directory name CL final classifications
        csvFilenameInfo  = results_dir + imagesSubDir + '-HI.csv' # directory name HI Hierarchical classifications
        if args.moviePredict != "": # Save results in a movie file 
            args.moviePredict = imagesSubDir + '.avi'  # use same name as csv file  
    else: # Process time-lapse images in directory
        imagesSubDir = args.images.split('/')[-2]
        if args.project == "MAMBO":
            # Format CSV result file names for project MAMBO
            imagesSubDir4 = args.images.split('/')[-4]
            imagesSubDir3 = args.images.split('/')[-3] 
            resultName = imagesSubDir4 + '-' + imagesSubDir3 + '-' + imagesSubDir
            csvFilename = results_dir + resultName + '-CL.csv' # directory name CL final classifications
            csvFilenameInfo  = results_dir + resultName +  '-HI.csv' # directory name HI Hierarchical classifications
            args.camera = imagesSubDir4 + '/' + imagesSubDir3 
            if args.moviePredict != "": # Save results in a movie file 
                args.moviePredict = resultName + '.avi'  # use same name as csv file   
        else:
            csvFilename = results_dir + imagesSubDir + '-CL.csv' # directory name CL final classifications
            csvFilenameInfo  = results_dir + imagesSubDir + '-HI.csv' # directory name HI Hierarchical classifications
            args.camera = imagesSubDir.split('_')[0] # first part is the name of the camera 
            if args.moviePredict != "": # Save results in a movie file 
                args.moviePredict = imagesSubDir + '.avi'  # use same name as csv file   
    
    # Create the CSV result file
    print(csvFilename)
    csvfile = open(csvFilename, 'w', newline = '\n')
    if args.CSVformat == "tracking":
        headerLine = "trap,trapId,date,time,taxaConf,taxaLabel,taxaId,taxaLevel,frameId,x1,y1,x2,y2,fileName\n"    
    else:
        headerLine = "year,trapDir,date,time,detectConf,detectId,x1,y1,x2,y2,fileName,taxaLabel,taxaId,taxaLevel,taxaConf,taxaSure,frameId\n"
        # Header line for tracking moths - includes species classifier
        #headerLine = "year,trap,date,time,detectConf,detectId,x1,y1,x2,y2,fileName,orderLabel,orderId,orderConf,aboveTH,key,speciesLabel,speciesId,speciesConf\n"
    csvfile.write(headerLine)
    csv_writer = csv.writer(csvfile, delimiter = ',')
    
    csvfileInfo = 0
    if type(modelClassifier) is HierarchicalClassifier:
        csvfileInfo = open(csvFilenameInfo, 'w', newline = '\n')
        headerLine = "Label1,LabelId1,Conf1,Above1,Label2,LabelId2,Conf2,Above2,Label3,LabelId3,Conf3,Above3,Checked,frameId\n"
        csvfileInfo.write(headerLine)

    saveMovie = False
    if args.moviePredict != "": # Save results in a movie file
        saveMovie = True
        scale = args.scale # Scale factor used when movie with predictions created
        dim = (int(imgWidth*scale), int(imgHeight*scale))
        movie_writer = cv2.VideoWriter(results_dir + args.moviePredict, cv2.VideoWriter_fourcc(*'DIVX'), fps, dim)

    frame_count = 0
    frames_after = 0
    if video_path != '': 
        # Processing video
        frame_time = start_time # frame_time global variable
        # Loop through the video frames
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()
            if success: 
                if (frame_count % frame_stride == 0):     
                            # Increment time based on fps = 1.0
                    frame_time = frame_time + datetime.timedelta(seconds=frame_stride)
                    prevFilname, frames_after = processFrame(frame, frame_time, frame_count, frames_after, useMotion, saveMovie, args, args.video.split('/')[-1], prevFilename)              
                frame_count += 1
            else:
                # Break the loop if the end of the video is reached
                break
        cap.release()
    else: 
        # Processing time-lapse images
        for image_file in sorted(os.listdir(args.images)):
            if image_file.endswith('.jpg') or image_file.endswith('.JPG'):
                if useMotion and prevFilename != '':
                    frame_time, dateTimeStr = getFrameTime(args.images, prevFilename.split('/')[-1], args.useExifTime)
                else:
                    frame_time, dateTimeStr = getFrameTime(args.images, image_file, args.useExifTime)
                #if (frame_count % frame_stride == 0):  
                print(image_file, dateTimeStr)
                frame = cv2.imread(args.images + image_file)
                #frame = cv2.resize(full_frame, (imgWidth, imgHeight), cv2.INTER_AREA) # Downsize image to HD size
                prevFilename, frames_after = processFrame(frame, frame_time, frame_count, frames_after, useMotion, saveMovie, args, imagesSubDir + '/' + image_file, prevFilename)              
                frame_count += 1

    # Release the video capture object and close the display window
    csvfile.close()
    if saveMovie:
        movie_writer.release()
    if csvfileInfo is not int:
        csvfileInfo.close()
