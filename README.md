# insectsDCT
This project contains Python code for processing time-lapse recorded images from insect camera traps. 
It contains code to detect, classify, and track insects with various backgrounds of plants and flowers.
(Detection, classification, and tracking, where floral cover estimation will be added later.)

Tracking should be used for high time-lapse recordings (0.33 - 1fps); otherwise, typical time-lapse intervals are 30 - 60 seconds.
Full-sized images are resized to 1920x1080 pixels for detection with YOLO11.
Insects detected with bounding boxes are cropped with rectangular windows and resized to 128x128 pixels for classification with CNN models.

## The algorithms used are described in the papers: 

Object detection with Motion-Informed Enhancement (MIE):

"Object Detection of Small Insects in Time-Lapse Camera Recordings".  <br />
https://www.mdpi.com/1424-8220/23/16/7242

Estimating flower cover with color and semantic segmentation and classification of 19 taxonomic groups of arthropods:

"A deep learning pipeline for time-lapse camera monitoring of floral environments and insect populations".  <br />
https://doi.org/10.1016/j.ecoinf.2024.102861 

Hierarchical classification of arthropods in three levels of taxonomic ranks:

"Hierarchical classification of insects with multitask learning and anomaly detection".  <br />
https://doi.org/10.1016/j.ecoinf.2023.102278

Tracking insects in low-framerate video recordings (<1fps):

"Towards edge processing of images from insect camera traps".  <br />
https://doi.org/10.1002/rse2.70007


# This repository includes the essential Python code for the steps in the figure below. 

![Alt text](PipelineHierarchical.png)

Estimating flower cover is currently in development. 

## Python environment files ##
README-conda-env-yolo11.txt - environment requirements

### Hierarchical model weights and labels ###

The modes below are trained on datasets collected with Wingscape's Bird cameras, Logitech Webcams, and Pi Cameras. 
Background images contain vegetation of Sedum, Red clover, Sea rocket, Common mallow, and different grasses.

V3. Third model (HierarchicalClassifierV3_05092025) trained on images recorded with cameras and backgrounds mentioned above and supplemented with GBIF data  <br />
https://drive.google.com/file/d/1zA22fWHYrmV-PKOHmddPX2OmHwxvxDb7/view?usp=drive_link

V4. Forth model (HierarchicalClassifierV4_05092025) trained on same images as in V3 but without GBIF data  <br />
https://drive.google.com/file/d/1ca2XaNygAE3UUUMkZtGWvmoy20AuaTHl/view?usp=sharing

Download the weights, labels, and thresholds from the above links. 
Save and unzip the file to the sub directory: insectsDCT/models_save

### Getting started ###

1. Download the repository and install it with the same directory structure.

2. Download weights, labels and thresholds for the hierachical classifier and unzip to: models_save/
   
4. Install the environment requirements see: README-conda-env-yolo11.txt (Anaconda)

5. Activate the python environment.

   - Anaconda: $ conda activate yolo11
  
6. Run the Python code to generate the CSV files for detection and tracking. (Sample images used - are found in: ./images)

   - $ python pipeDetectAndClassifyInsectsTaxon.py  <br />
     Performs detection and classification with ResNet50 on CUDA:0
     
   - $ python pipeDetectAndClassifyInsectsTaxon.py --device cpu  <br />
	 Performs detection and classification with ResNet50 on CPU
     
   - $ python pipeDetectAndClassifyInsectsTaxon.py --modelType ConvNextBase  <br />
     Performs detection and classification using ConvNextBase model (best performing model)
     
   - $ python pipeDetectAndClassifyInsectsTaxon.py --dataset V4  <br />
     Performs detection and classification using models trained on classification dataset V4 instead of V3
     
   - $ python pipeTrackInsectsTaxon.py  <br />
     Performs tracking based on the CSV output files (./detections/*-CL.csv)
   
   See code for additional parameters for the above python script.

7. Run the Python code to generate images of insect crops based on taxa classification and tracking described in 6.
   
   - $ python createCrops.py --CSVfiles "./detections/" --imagesPath "./images/" --cropsPath "./crops/"  <br />
     Creates cropped images of detected and classified insects sorted to directories based on *-CL.csv files
       
   - $ python createTrackCrops.py --validConfTH 20  --tracks "./tracks" --images "./images" --resultsDir "./trackCrops" <br /> 
     Creates plots of tracks with cropped images of classified insects sorted to directories based on *-TRS.csv files

To use a simple flat classifier with few classes of taxons see description in: 

https://github.com/kimbjerge/insectsDCT/tree/main/README-flat.md


### CSV files in detections directory ###

Content of *-CL.csv files which contain lines for each detection (subdir3-subdir2-subdir1-CL.csv):

	year,trapDir,date,time,detectConf,detectId,x1,y1,x2,y2,fileName,taxaLabel,taxaId,taxaLevel,taxaConf,taxaSure,frameId

- trapDir is the directory of the source files subdir3/subdir2  <br />
- detectConf is the confidence score of the YOLO11 arthropod detector  <br />
- detectId is always 0 and is the class id determined by YOLO11  <br />
- x1,y1,x2,y2 is the coordinates in the images for the upper left corner and lower right corner of the bounding box surrounding the detected arthropod  <br />
- filename is the name of the image file with the format subdir1/name.JPG  <br />
- taxaLabel is the order, family, genus or species name of the arthropod found by the hierarchical classifier see names below (taxaLevel 1-3)
 
- taxaId and taxaLevel will be updated with the following classification codes

Hierarchical taxa of classes in the model HierarchicalClassifierV4_05092025:

taxaLevel 1: (21 groups of taxa primary Order)

    1. Apoidea
    2. Aranaea 
    3. Birds 
    4. Coleoptera
    5. Dermaptera
    6. Diptera
    7. Feathers
    8. Frogs
    9. Hemiptera
    10. Hymenoptera
    11. Isopoda
    12. Larvae
    13. Lepidoptera
    14. Lepidoptera_fw
    15. Lizards
    16. Milipedes
    17. Odonata
    18. Orthoptera
    19. Slugs
    20. Snails
    21. Vegetation

taxaLevel 2: (43 groups of taxa primary Family - some of the below labels do also contain level 1 names)

    1. Acrididae
    2. Andrenidae
    3. Apidae
    4. Apoidea
    5. Apoidea_small
    6. Aranaea
    7. Birds
    8. Bombyliidae
    9. Cantharidae
    10. Coccinellidae
    11. Coleoptera
    12. Crabronidae
    13. Dermaptera
    14. Diptera
    15. Feathers
    16. Formidicidae
    17. Frogs
    18. Halictidae
    19. Hemiptera
    20. Hesperidae
    21. Ichneumonidae
    22. Isopoda
    23. Larvae
    24. Lepidoptera
    25. Lepidoptera_fw
    26. Lizards
    27. Lycaenidae
    28. Megachilidae
    29. Milipedes
    30. Moths
    31. Nymphalidae
    32. Nymphalidae_fw
    33. Odonata
    34. Pieridae
    35. Pompilidae
    36. Sarcophagidae
    37. Slugs
    38. Snails
    39. Syrphidae
    40. Tetrigidae
    41. Tettigoniidae
    42. Vegetation
    43. Vespidae

taxaLevel 3: (83 groups of taxa primary Genus or Species - some of the below labels do also contain level 1+2 names)

    1. Acrididae
    2. Aglais io
    3. Aglais urticae
    4. Aglais urticae_fw
    5. Ancistrocerus nigricornis
    6. Andrena red_abdomen
    7. Andrena reddish
    8. Aphantopus hyperantus
    9. Apis mellifera     
    10. Apoidea            
    11. Apoidea_small      
    12. Aranaea            
    13. Birds              
    14. Bombus             
    15. Bombus hypnorum    
    16. Bombus lapidarius  
    17. Bombus pascuorum   
    18. Bombus sylvarum    
    19. Bombus terrestris  
    20. Bombyliidae        
    21. Chorthippus        
    22. Chorthippus apricarius
    23. Chrysotoxum           
    24. Coccinella septempunctata
    25. Coenonympha pamphilus    
    26. Coleoptera               
    27. Crabronidae              
    28. Decticus verrucivorus    
    29. Dermaptera               
    30. Diptera                  
    31. Episyrphus balteatus     
    31. Eristalis                
    31. Eumenes coronatus        
    31. Eupeodes                 
    31. Eurimyia                 
    31. Feathers                 
    31. Formidicidae             
    31. Frogs                    
    31. Halictidae striped       
    40. Helophilus               
    41. Hemiptera                
    42. Hesperidae               
    43. Ichneumonidae            
    44. Isopoda                  
    45. Larvae                   
    46. Lasiommata megera        
    47. Lepidoptera              
    48. Lepidoptera_fw           
    49. Lizards                  
    50. Lycaena phlaeas          
    51. Lycaenidae               
    52. Maniola jurtina          
    53. Maniola jurtina_fw       
    54. Megachilidae             
    55. Meliscaeva cinctella     
    56. Melitaea cinxia          
    57. Milipedes                
    58. Moths                    
    59. Myathropa florea         
    60. Odonata                  
    61. Omocestus viridulus      
    62. Pholidoptera griseoaptera
    63. Pieridae                 
    64. Platycheirus             
    65. Pompilidae               
    66. Rhagonycha fulva         
    67. Sarcophagidae            
    68. Satyrinae                
    69. Satyrinae_fw             
    70. Scaeva pyrastri          
    71. Scaeva selenitica        
    72. Slugs                    
    73. Snails                   
    74. Sphaerophoria scripta-complex
    75. Sphecodes                    
    76. Syritta pipiens              
    77. Syrphidae                    
    78. Syrphus                      
    79. Tetrix                       
    80. Vegetation                   
    81. Vespidae                     
    82. Vespula vulgaris             
    83. Xanthogramma                 

- taxaConf is the confidence score from the classifier (0-100%)  <br />
- taxaSure is True if the confidence score is within the distribution of the training data and the classification is valid  <br />
- frameId is a number used to find the corresponding entry in the *-HI.csv file with more detailed information of the hierarchical classifier  <br />

Example of *-CL.csv content:

	2024,FR02_Bagnas/Cam.A.2024.07.02,20240407,141529,48,0,2018,212,2112,270,101_WSCT/WSCT0441.JPG,Unsure,0,0,0,False,441
	2024,FR02_Bagnas/Cam.A.2024.07.02,20240407,141629,43,0,1262,1123,1355,1224,101_WSCT/WSCT0442.JPG,Syrphidae,37,2,6,True,442
	2024,FR02_Bagnas/Cam.A.2024.07.02,20240407,142529,62,0,778,1650,893,1716,101_WSCT/WSCT0451.JPG,Sphaerophoria scripta-complex,73,3,45,True,451 

Content of *-HI.csv files which contain hierarchical taxonomic information for each detection (subdir3-subdir2-subdir1-CL.csv):

	Label1,LabelId1,Conf1,Above1,Label2,LabelId2,Conf2,Above2,Label3,LabelId3,Conf3,Above3,Checked,frameId

LabelX, LabelIdX is ConfX is the name, id, confidence score on the taxonomic levels 1, 2 and 3.  <br />
AboveX is True if the confidence scores is within the distribution of the training data. <br />
Checked is True if the classification is correct according to the dependences in the taxonomic hierarchy.

Example of content with same frameId as in example above (*-CL.csv content):

	Hymenoptera,7,0.0016508539215075745,False,Halictidae,14,0.05190288999028045,True,Sphaerophoria scripta-complex,72,0.003527139374961427,False,False,441
	Diptera,3,0.06632232090118803,True,Syrphidae,36,0.05682376443188935,True,Sphaerophoria scripta-complex,72,0.00486999349185118,False,True,442
	Diptera,3,0.47108301131483055,True,Syrphidae,36,0.38675071058276406,True,Sphaerophoria scripta-complex,72,0.4498128146023822,True,True,451

 ### CSV and JSON files in tracks directory ###

Content of *.csv files which contain lines for each track (piX_YYYY_MM_DD-TR.csv):

	id,startdate,starttime,endtime,duration,class,counts,confidence,size,distance

Where class is the name of the taxonId at level 1-3 and id is the track number

Example:

	0,20250221,11:57:31,11:58:23,52,Megachile,18,36.8,3199,3171  
	1,20250221,11:58:07,11:58:31,24,Megachile,12,53.9,2686,1364

counts is the number of detections in a track (at least two detections to make a track)  <br />
confidence is the number of times the class was predicted relative to all detections in the track   <br />
size is the average pixel size of the tracked insect   <br />
distance is the distance in pixels the insect was tracked   <br />

Content of *.csv files which contain lines for each detection in each track (piX_YYYY_MM_DD-TRS.csv):

	id,date,time,taxaConf,taxaLabel,xc,yc,x1,y1,width,height,detectLine,fileName

Example:

	0,20250221,115731,0.0,Unsure,1331,632,1307,600,49,64,1,pi2_2025_02_21_11_57_31.jpg
	0,20250221,115732,21.13,Megachile,1310,674,1285,640,50,68,2,pi2_2025_02_21_11_57_32.jpg
	0,20250221,115734,0.72,Andrena,1278,700,1252,682,52,37,3,pi2_2025_02_21_11_57_34.jpg

taxaConf is the taxa confidence score same as confidence in the detection CSV file   <br />
detectLine is the line number in the detection CSV file


## Training insect detector and hierachical classification models ##

Datasets for detector and classifier is not part of this Github repository. (Will be published later)

### Code for inspiration to create datasets with motion (MIE) images: ###

  - createAccurateDataset.py, Create_MotionNI-dataset.py

### Subdirectories with python helper classes ###

  - common - contains Python code used by pipeDetectAndClassifyInsectsTaxon.py  <br />
  - idac - contains python code used by pipeTrackInsectsTaxon.py

### Training and validating models (YOLO11) on color or motion images: ###

 - insectsColorTrain.py, insectsColorVal.py  <br />
 - insectsMotionTrain.py

### Training and validating hierarchical classification models: ###

 - hierarchicalB3L/trainAdv.py  traning of the hierarchical classifier<br />
 - hierarchicalB3L/validate.py  validation of  the hierarchical classifier<br />
 - hierarchicalB3L/plotTrainAdvResults.py plotting results and learning the distribution (mean+STD) of logits for each class in the hierarchy

## Additional helper functions ##

 - plotResultsMAMBO.py and plotResultsNI2.py - examples of reading CSV files with detections and classifications and plots histograms of taxa of arthoropods
   
### Helper functions to create datasets for the insect detector model ###

 - countLabels.py - counts the labels on the datasets for YOLO detector for images with and without annotations
 - Create-MotionNI-dataset.py - example for how to create motion enhanced images (MIE)
 - Create-PollNI-dataset.py - selects and create dataset for detection based on images from project Pollinator Watch (Logitech camera)
 - Create-Orchard-dataset.py - selects and create dataset for detection based on image from project Orchard (Pi model 3 camera)
 - createAccurateDataset.py - dataset from paper: "Accurate detections and identification .." https://doi.org/10.1371/journal.pstr.0000051 

More information on the dataset for insect detection: 
  
https://github.com/kimbjerge/insectsDCT/tree/main/README_detection_datasets.txt

### Helper functions to select crops (see createCrops.py) for dataset to improve training of the hierarchical classifier ###

 - copySelectedCrops.py - example code for how to select specific taxa of insect crops and create a dataset for training (MAMBO, NI2)

### Helper functions to evaluate the hierarchical classifier and tracking ###

 - createCrops.py - creates crops of insect images found by the insect detector and classifier
 - countCrops.py - counts the created crops and plots statistics for number of FalseA (subdirectory of false arthropods) and FalseB (subdirectory of false background detections)
 - createTrackCrops.py - based on the *_TRS.csv files with valid tracks (Id) - it create png files of insects crops in a 4x4 matrix plot (See example below)

   ![Alt text](ExTrackCrops.png)
   
