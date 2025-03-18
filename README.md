# insectsDCT
This project contains Python code for processing time-lapse (0.2-1fps) images (resized to 1920x1080 pixel) from insect camera traps. 
Code to detect, classify, and track insects with a background of plants and flowers.
(detection, classification, tracking, and floral cover estimation)

## The algorithms used are described in the papers: 

Object detection with Motion-Informed Enhancement (MIE):

"Object Detection of Small Insects in Time-Lapse Camera Recordings".  <br />
https://www.mdpi.com/1424-8220/23/16/7242

Estimating flower cover with color and semantic segmentation and classification of 19 taxonomic groups of arthropods:

"A deep learning pipeline for time-lapse camera monitoring of floral environments and insect populations".  <br />
https://doi.org/10.1016/j.ecoinf.2024.102861 

Tracking insects in low-framerate video recordings (<1fps):

"Towards edge processing of images from insect camera traps".  <br />
https://www.biorxiv.org/content/10.1101/2024.07.01.601488v2


# This repository includes the essential Python code for the steps in the figure below. 

![Alt text](Pipeline.png)

Estimating flower cover is currently in development. 

## Python environment files ##
README-conda-env-yolo11.txt - environment requirements

### Getting started ###

1. Download the repository and install it with the same directory structure

2. Download the weights for the classifier and unzip to: models_save/EfficientNetB4-19cls-50-Ext-Finetuned.keras
   
https://drive.google.com/file/d/1_zjJS_Y5aIr2OFN6Jmg9aRAtILV0LV0J/view?usp=sharing
   
4. Install the environment requirements see: README-conda-env-yolo11.txt (Anaconda)

5. Activate the python environment

   - Anaconda: $ conda activate yolo11
  
6. Run the Python code to generate the CSV files for detection and tracking

   - $ python pipeDetectAndClassifyInsects.py (Performs detection and classification on CUDA:0)
   - $ python pipeDetectAndClassifyInsects.py --device cpu (Performs detection and classification on CPU)
   - $ python pipeTrackInsects.py (Performs tracking based on the CSV output files) 

### CSV files in detections directory ###

Content of *.csv files which contain lines for each detection (piX_YYYY_MM_DD.csv):

	trap,trapId,date,time,orderConf,orderId,x1,y1,x2,y2,fileName

Where the orderId will be updated with the following classification codes:

	1 - Coccinellidae (Ladybirds) 
	2 - Coleoptera (Beetles) 
	3 - Background (Plants)
	4 - Bombus (Bumblebees)
	5 - Syrphidae (Hoverflies) 
	6 - Lepidoptera (Butterflies)
	7 - Araneae (Spiders)
	8 - Formidicidae (Ants)
	9 - Diptera (Flies)
	10 - Hemiptera (True bugs)
	11 - Isopoda (Isopods)
	12 - Unspecified 
	13 - Hymenoptera
	14 - Orthoptera (Grasshoppers)
	15 - Rhagnoycha fulva
	16 - Satyrinae (Satyrines)
	17 - Aglais urticea (Small tortoiseshell)
	18 - Odonata (Dragonflies)
	19 - Apis mellifera (Honeybees)

orderConf is the confidence score from the classifier (0-100%)

Example:

	pi1,1,20250221,115753,64,19,1335,820,1386,888,pi1_2025_02_21/pi2_2025_02_21_11_57_53.jpg 
	pi1,1,20250221,115807,39,12,422,729,489,776,pi1_2025_02_21/pi2_2025_02_21_11_58_07.jpg

### CSV and JSON files in tracks directory ###

Content of *.csv files which contain lines for each track (piX_YYYY_MM_DDTR.csv):

	id,startdate,starttime,endtime,duration,class,counts,confidence,size,distance

Where class is the name of the orderId and id is the track number

Example:

	0,20250221,11:57:31,11:58:23,52.00,Hymenoptera,18,36.84,3198.74,3171  
	1,20250221,11:58:07,11:58:31,24.00,Hymenoptera,12,53.85,2686.08,1364

counts is the number of detections minus one in a track (at least two detections to make a track)  <br />
confidence is the number of times the class was predicted relative to all detections in the track   <br />
size is the average pixel size of the tracked insect   <br />
distance is the distance in pixels the insect was tracked   <br />

Content of *.csv files which contain lines for each detection in each track (piX_YYYY_MM_DDTRS.csv):

	id,date,time,percent,class,xc,yc,x1,y1,width,height,image

Example:

	0,20250221,115731,60,Hymenoptera,1331,632,1307,600,49,64,pi2_2025_02_21_11_57_31.jpg  
	0,20250221,115732,79,Hymenoptera,1310,674,1285,640,50,68,pi2_2025_02_21_11_57_32.jpg 
	0,20250221,115734,63,Background,1278,700,1252,682,52,37,pi2_2025_02_21_11_57_34.jpg

percent is the confidence score same as confidence in the detecion CSV file

## Training insect detector and classifier models ##

### Code for inspiration to create datasets with motion (MIE) images: ###

  - createAccurateDataset.py, Create_MotionNI-dataset.py

### Subdirectories with python helper classes ###

  - common - contains Python code used by pipeDetectAndClassifyInsects.py  <br />
  - idac - contains python code used by pipeTrackInsects.py

### Training and validating models (YOLO11) on color or motion images: ###

 - insectsColorTrain.py, insectsColorVal.py  <br />
 - insectsMotionTrain.py

## Training and validation of arthropod classifier (EfficientNetB4) ##

### Dataset for training the arthropod classifier (NI2-19cls) can be downloaded from Zenodo ###
https://zenodo.org/records/13772695
Copy the NI2-19cls.zip to datasets and unpack the file

### Training arthropod classifier with 19 taxonomic groups ###

 - training-ClassificationExtended19Cls.py - used for training SOAT CNNs including: ResNetv50, EfficientNetB4, MobileNetv2, InceptionV3, DenseNet201, ConvNeXtTiny and ConvNeXtBase

