datasets:

==============
accurateColor
accurateMotion
==============

Paper: Accurate detection and identification of insects from camera trap images with deep learning
https://journals.plos.org/sustainabilitytransformation/article?id=10.1371/journal.pstr.0000051

Subset of dataset: https://zenodo.org/records/7395752

    # Coccinellidae septempunctata  0  -> each 4th
    # Apis mellifera                1  -> none
    # Bombus lapidarius             2  -> all
    # Bombus terrestris             3  -> each 2nd
    # Eupeodes corolla              4  -> each 2nd
    # Episyrphus balteatus          5  -> all
    # Aglais urticae                6  -> all   
    # Vespula vulgaris              7  -> all
    # Eristalis tenax               8  -> all
    # [1629, 588, 1536, 1480, 1839, 1267, 282, 952, 285]  labels total: 9858
    # Background images: 1153 (574 lesser)
    # Images total: 9959 (9385)

Dataset	Insects	Images	Background
Train	9441	9385	980
Val	3151	2453	37

==============
beesColor
beesMotion
==============

Paper: Object Detection of Small Insects in Time-Lapse Camera Recordings
https://www.mdpi.com/1424-8220/23/16/7242
Dataset: https://vision.eng.au.dk/mie/

Dataset	Insects	Images	Background
Train	2499	3783	1953
Val	568	946	508
Total	3067	4729	2461

==============
insects2Color
insects2Motion
==============
Combines dataset beesColor and beesMotion and arthropods from:
/home/don/yolov5r/datasets/insects/images/trainI21 (Color)
/home/don/yolov5r/datasets/insects/images/trainI21m (Motion)

published in:

Paper: A deep learning pipeline for time-lapse camera monitoring of insects and their floral environments
https://www.sciencedirect.com/science/article/pii/S1574954124004035
Dataset: https://zenodo.org/records/13772695

Dataset	Images	Arthropods	Background
Training 6,901	     4,770	      2654
Testing	   766	       525	       299

Combined dataset insects2:
-------------------------
Dataset	Insects	Images	Background
Train	7269	10684	4607
Val	1093	1712	807

==============
insects3Color
insects3Motion
==============
Combines dataset accurateColor and accurateMotion and insects2

Combined dataset insects3:
-------------------------
Dataset	Insects	Images	Background
Train	16710	20069	5587
Val	4244	4165	844


===========
countLabels:
===========

Accurate train:  [9441, 0, 0, 0, 0, 0, 0, 0, 0]  total: 9441
Total images: 9385 background: 980
Bees train:  [2499, 0, 0, 0, 0, 0, 0, 0, 0]  total: 2499
Total images: 3783 background: 1953
Arthropods train:  [4770, 0, 0, 0, 0, 0, 0, 0, 0]  total: 4770
Total images: 6901 background: 2654
Insects2 train:  [7269, 0, 0, 0, 0, 0, 0, 0, 0]  total: 7269
Total images: 10684 background: 4607
Insects3 train:  [16710, 0, 0, 0, 0, 0, 0, 0, 0]  total: 16710
Total images: 20069 background: 5587

Accurate val:  [3151, 0, 0, 0, 0, 0, 0, 0, 0]  total: 3151
Total images: 2453 background: 37
Bees val:  [568, 0, 0, 0, 0, 0, 0, 0, 0]  total: 568
Total images: 946 background: 508
Arthropods val:  [525, 0, 0, 0, 0, 0, 0, 0, 0]  total: 525
Total images: 766 background: 299
Insects2 val:  [1093, 0, 0, 0, 0, 0, 0, 0, 0]  total: 1093
Total images: 1712 background: 807
Insects3 val:  [4244, 0, 0, 0, 0, 0, 0, 0, 0]  total: 4244
Total images: 4165 background: 844


================
Pollinator Watch:
27/9 - 2025
================

Selected images with: insectsDCT/Create-PollNI-dataset.py

Using random selected images from camera systems S2, S3 and S4 (collected in 2020)
Selected random images based on hierarchical classifier with detector V3 and classifier V3:

3x600 images that contain insect
3x300 images that contain unsure classifications
3x300 images that contain vegetation

20% selected for validation

Pollinator Watch train:  [468, 0, 0, 0, 0, 0, 0, 0, 0]  total: 468
Total images: 2867 background: 2488
Pollinator Watch val:  [118, 0, 0, 0, 0, 0, 0, 0, 0]  total: 118
Total images: 720 background: 625

================
UFZ Orchard:
29/9 - 2025
================

Selected images with: insectsDCT/Create-Orchard-dataset.py

Using random selected images from detections collected 2025 in: /UFZ/detectionsTrain/ (Pi8, Pi10, Pi11, Pi12, Pi15, Pi20, Pi26 and Pi28 - total 9 days)
Some sites with artificial background (Pi8, Pi11, Pi20, Pi26, Pi28) 
Selected random images based on hierarchical classifier with detector V3 and classifier V3:

2500 images that contain insect
500 images that contain unsure classifications
1000 images that contain vegetation

20% selected for validation

UFZ Orchard train:  [2705, 0, 0, 0, 0, 0, 0, 0, 0]  total: 2705
Total images: 3157 background: 730
UFZ Orchard val:  [696, 0, 0, 0, 0, 0, 0, 0, 0]  total: 696
Total images: 798 background: 173

================
MAMBO:
29/9 - 2025
================

Selected images with: insectsDCT/Create-MAMBO-dataset.py

Using random selected images from detections collected in 2024: /MAMBO/ (au-DK, cirad-FR, ecoinn-MT, ufz-DE)
Selected random images based on hierarchical classifier with detector V3 and classifier V3:

4000 images that contain insect
1000 images that contain unsure classifications
1000 images that contain vegetation

20% selected for validation

MAMBO train:  [5752, 0, 0, 0, 0, 0, 0, 0, 0]  total: 5752
Total images: 4313 background: 806
MAMBO val:  [1635, 0, 0, 0, 0, 0, 0, 0, 0]  total: 1635
Total images: 1163 background: 206

Comments to dataset V5:
======================

Ignoring animals including slugs and snails.(but not lizzards and frogs)

6 different projects: NatureImpact I and NatureImpact II, Greenhouse, Pollinator Watch, UFZ Orcard, MAMBO
3 different cameras: Wingscapes, Logitech webcamera, Pi model 3 camera.
7 different backgrounds: Sedum, Clover, Sea rocket, Common Mallow, Mixed Wild Grasses and flowers, Malta flowers, Artificial flowers

