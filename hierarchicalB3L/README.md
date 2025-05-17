# insectsDCT - Arthropod detector with hierarchical taxon classifier
This project contains Python code for processing time-lapse recorded images (30-60 sec. intervals, resized to 1920x1080 pixel) from insect camera traps. 
Code to detect, classify arthropods.

## The algorithms used are described in the papers: 

Object detection with Motion-Informed Enhancement (MIE):

"Object Detection of Small Insects in Time-Lapse Camera Recordings".  <br />
https://www.mdpi.com/1424-8220/23/16/7242

Hierarchical classification of arthropods in three levels of taxonomic ranks:

"Hierarchical classification of insects with multitask learning and anomaly detection".  <br />
https://doi.org/10.1016/j.ecoinf.2023.102278

  
1. Run the Python code to generate the CSV files for detection and classification

   - $ python pipeDetectAndClassifyInsectsTaxon.py (Performs detection and classification on CUDA:0)
   - $ python pipeDetectAndClassifyInsectsTaxon.py --device cpu (Performs detection and classification on CPU)

### CSV files in detections directory ###

Content of *-CL.csv files which contain lines for each detection (subdir3-subdir2-subdir1-CL.csv):

	year,trapDir,date,time,detectConf,detectId,x1,y1,x2,y2,fileName,taxaLabel,taxaId,taxaLevel,taxaConf,taxaSure,frameId

- trapDir is the directory of the source files subdir3/subdir2  <br />
- detectConf is the confidence score of the YOLO11 arthropod detector  <br />
- detectId is always 0 and is the class id determined by YOLO11  <br />
- x1,y1,x2,y2 is the coordinates in the images for the upper left corner and lower right corner of the bounding box surrounding the detected arthropod  <br />
- filename is the name of the image file with the format subdir1/name.JPG  <br />
- taxaLabel is the order, family, genus or species name of the arthropod found by the hierarchical classifier see names below (taxaLevel 1-3)
 
- taxaId and taxaLevel will be updated with the following classification codes:

taxaLevel 1: (Order)

	1   Aranaea
	2   Birds
	3   Coleoptera
	4   Diptera
	5   Feathers
	6   Frogs
	7   Hemiptera
	8   Hymenoptera
	9   Isopoda
	10   Larvae
	11   Lepidoptera
	12   Lepidoptera_fw
	13   Lizards
	14   Milipedes
	15   Odonata
	16   Orthoptera
	17   Slugs
	18   Snails
	19   Vegetation

taxaLevel 2: (Family - some of the below labels do also contain level 1 names)

	1   Acrididae
	2   Andrenidae
	3   Apidae
	4   Aranaea
	5   Birds
	6   Cerambycidae
	7   Coccinellidae
	8   Coleoptera
	9   Colletidae
	10   Crabronidae
	11   Diptera
	12   Feathers
	13   Formidicidae
	14   Frogs
	15   Halictidae
	16   Hemiptera
	17   Hesperidae
	18   Ichneumonidae
	19   Isopoda
	20   Larvae
	21   Lepidoptera
	22   Lepidoptera_fw
	23   Lizards
	24   Lycenidae
	25   Megachilidae
	26   Milipedes
	27   Moths
	28   Nymphalidae
	29   Nymphalidae_fw
	30   Odonata
	31   Pieridae
	32   Pompilidae
	33   Satyrinae
	34   Satyrinae_fw
	35   Slugs
	36   Snails
	37   Syrphidae
	38   Tetrigidae
	39   Tettigoniidae
	40   Vegetation
	41   Vespidae


taxaLevel 3: (Genus or Species - some of the below labels do also contain level 1+2 names)

	1   Acrididae
	2   Aglais io
	3   Aglais urticae
	4   Aglais urticae_fw
	5   Aglias io
	6   Ancistrocerus nigricornis
	7   Andrena
	8   Aphantopus hyperantus
	9   Apis mellifera
	10   Aranaea
	11   Birds
	12   Bombus
	13   Bombus hypnorum
	14   Bombus lapidarius
	15   Bombus pascuorum
	16   Bombus terrestris
	17   Chorthippus
	18   Chorthippus apricarius
	19   Chrysotoxum
	20   Coccinella septempunctata
	21   Coenonympha pamphilus
	22   Coleoptera
	23   Crabronidae
	24   Decticus verrucivorus
	25   Diptera
	26   Episyrphus balteatus
	27   Eristalis
	28   Eristalis tenax
	29   Eumenes coronatus
	30   Eupeodes
	31   Eupeodes corollae
	32   Eurimyia
	33   Feathers
	34   Formidicidae
	35   Frogs
	36   Halictus
	37   Helophilus
	38   Hemiptera
	39   Hesperidae
	40   Hylaeus
	41   Ichneumonidae
	42   Isopoda
	43   Larvae
	44   Lasioglossum
	45   Lepidoptera
	46   Lepidoptera_fw
	47   Lizards
	48   Lycaena phlaeas
	49   Lycenidae
	50   Maniola jurtina
	51   Maniola jurtina_fw
	52   Megachile
	53   Meliscaeva cinctella
	54   Melitaea cinxia
	55   Milipedes
	56   Moths
	57   Myathropa florea
	58   Nomada
	59   Odonata
	60   Omocestus viridulus
	61   Osmia
	62   Pholidoptera griseoaptera
	63   Pieridae
	64   Platycheirus
	65   Pompilidae
	66   Rhagonycha fulva
	67   Satyrinae
	68   Satyrinae_fw
	69   Scaeva pyrastri
	70   Scaeva selenitica
	71   Slugs
	72   Snails
	73   Sphaerophoria scripta-complex
	74   Sphecodes
	75   Syritta pipiens
	76   Syrphidae
	77   Syrphus
	78   Tetrix
	79   Vegetation
	80   Vespula vulgaris


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
Checked is True if the classification is correct according to the dependences in the taxanomic hierarchy.

Example of content with same frameId as in example above (*-CL.csv content):

	Hymenoptera,7,0.0016508539215075745,False,Halictidae,14,0.05190288999028045,True,Sphaerophoria scripta-complex,72,0.003527139374961427,False,False,441
	Diptera,3,0.06632232090118803,True,Syrphidae,36,0.05682376443188935,True,Sphaerophoria scripta-complex,72,0.00486999349185118,False,True,442
	Diptera,3,0.47108301131483055,True,Syrphidae,36,0.38675071058276406,True,Sphaerophoria scripta-complex,72,0.4498128146023822,True,True,451
	