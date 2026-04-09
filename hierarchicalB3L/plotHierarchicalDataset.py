# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 13:28:48 2025

@author: Kim Bjerge
"""

import matplotlib.pyplot as plt
import numpy as np
from hierarchical_loader import HierarchicalDatasetLoader


"""
# Dataset V2
level1 = {
'Aranaea':[                1658,    184],
'Birds':[                    74,      9],
'Coleoptera':[             7382,    820],
'Diptera':[                6991,    777],
'Feathers':[                154,     17],
'Frogs':[                    16,      1],
'Hemiptera':[               576,     64],
'Hymenoptera':[           15472,   1719],
'Isopoda':[                 767,     85],
'Larvae':[                 172,     19],
'Lepidoptera':[            2608,    291],
'Lepidoptera_fw':[         4743,    527],
'Lizards':[                  23,      2],
'Milipedes':[                69,      8],
'Mixed':[                  1390,    154],
'Odonata':[                 456,     51],
'Orthoptera':[             1300,    144],
'Slugs':[                   971,    108],
'Snails':[                  152,     17],
'Vegetation':[             4942,    549]
}

level2 = {
'Acrididae':[              1240,    138],
'Andrenidae':[               49,      6],
'Apidae':[                13567,   1507],
'Aranaea':[                1658,    184],
'Birds':[                    74,      9],
'Cerambycidae':[            740,     83],
'Coccinellidae':[          6312,    701],
'Coleoptera':[              330,     36],
'Colletidae':[               49,      5],
'Crabronidae':[               5,      1],
'Diptera':[                 608,     68],
'Feathers':[                154,     17],
'Formidicidae':[            547,     61],
'Frogs':[                    16,      1],
'Halictidae':[               43,      5],
'Hemiptera':[              576 ,    64],
'Hesperidae':[              107,     12],
'Hymenoptera':[              38,      4],
'Ichneumonidae':[            17,      2],
'Isopoda':[                 767,     85],
'Larvae':[                  172,     19],
'Lepidoptera':[             140,     16],
'Lepidoptera_fw':[          233,     25],
'Lizards':[                  23,      2],
'Lycenidae':[                43,      5],
'Megachilidae':[             85,      9],
'Milipedes':[                69,      8],
'Mixed':[                  1390,    154],
'Moths':[                    38,      5],
'Nymphalidae':[            2235,    248],
'Nymphalidae_fw':[         3361,    374],
'Odonata':[                 456,     51],
'Orthoptera':[                4,      0],
'Pieridae':[                 35,      4],
'Pompilidae':[                6,      1],
'Satyrinae':[                10,      1],
'Satyrinae_fw':[           1149,    128],
'Slugs':[                   971,    108],
'Snails':[                  152,     17],
'Syrphidae':[              6383,    709],
'Tetrigidae':[                4,      1],
'Tettigoniidae':[            52,      5],
'Vegetation':[             4942,    549],
'Vespidae':[               1066,    118]
}

level3 = {
'Acrididae':[                         903,    100],
'Aglais io':[                          47,      5],
'Aglais urticae':[                   1605,    178],
'Aglais urticae_fw':[                 316,     36],
'Aglias io':[                           5,      0],
'Ancistrocerus nigricornis':[          11,      1],
'Andrena':[                            49,      6],
'Aphantopus hyperantus':[              32,      4],
'Apis mellifera':[                   8513,    946],
'Aranaea':[                          1658,    184],
'Birds':[                              74,      9],
'Bombus':[                            187,     20],
'Bombus hypnorum':[                    18,      2],
'Bombus lapidarius':[                1912,    213],
'Bombus pascuorum':[                   12,      1],
'Bombus terrestris':[                2923,    325],
'Chorthippus':[                       285,     32],
'Chorthippus apricarius':[             29,      3],
'Chrysotoxum':[                         4,      0],
'Coccinella septempunctata':[        6312,    701],
'Coelioxys':[                           1,      0],
'Coenonympha pamphilus':[              69,      8],
'Coleoptera':[                        330,     36],
'Crabronidae':[                         5,      1],
'Decticus verrucivorus':[               2,      0],
'Diptera':[                           608,     68],
'Episyrphus balteatus':[             2032,    226],
'Eristalis':[                           4,      1],
'Eristalis tenax':[                   283,     31],
'Eumenes coronatus':[                   3,      0],
'Eupeodes':[                          339,     37],
'Eupeodes corollae':[                3316,    369],
'Eurimyia':[                           50,      6],
'Feathers':[                          154,     17],
'Formidicidae':[                      547,     61],
'Frogs':[                              16,      1],
'Halictus':[                            6,      1],
'Helophilus':[                          5,      0],
'Hemiptera':[                         576,     64],
'Hesperidae':[                        107,     12],
'Hylaeus':[                            49,      5],
'Hymenoptera':[                        38,      4],
'Ichneumonidae':[                      17,      2],
'Isopoda':[                           767,     85],
'Larvae':[                            172,     19],
'Lasioglossum':[                       34,      3],
'Lepidoptera':[                       140,     16],
'Lepidoptera_fw':[                    233,     25],
'Lizards':[                            23,      2],
'Lycaena phlaeas':[                    37,      4],
'Lycenidae':[                           6,      1],
'Maniola jurtina':[                   476,     53],
'Maniola jurtina_fw':[               3045,    338],
'Megachile':[                          81,      9],
'Meliscaeva cinctella':[                8,      1],
'Melitaea cinxia':[                     1,      0],
'Milipedes':[                          69,      8],
'Mixed':[                            1390,    154],
'Moths':[                              38,      5],
'Myathropa florea':[                    1,      0],
'Nomada':[                              2,      0],
'Odonata':[                           456,     51],
'Omocestus viridulus':[                23,      3],
'Orthoptera':[                          4,      0],
'Osmia':[                               3,      0],
'Pholidoptera griseoaptera':[          50,      5],
'Pieridae':[                           35,      4],
'Platycheirus':[                        7,      1],
'Pompilidae':[                          6,      1],
'Rhagonycha fulva':[                  740,     83],
'Satyrinae':[                          10,      1],
'Satyrinae_fw':[                     1149,    128],
'Scaeva pyrastri':[                     5,      1],
'Scaeva selenitica':[                   7,      1],
'Slugs':[                             971,    108],
'Snails':[                            152,     17],
'Sphaerophoria scripta-complex':[     187,     20],
'Sphecodes':[                           3,      1],
'Syritta pipiens':[                    12,      2],
'Syrphidae':[                         113,     12],
'Syrphus':[                            10,      1],
'Tetrix':[                              4,      1],
'Vegetation':[                       4942,    549],
'Vespula vulgaris':[                 1052,    117]
}

"""

# Classification dataset V6
image_path_list = ['F:/insectsDCT_datasets/classifier/datasetV6/NI2',
                   'F:/insectsDCT_datasets/classifier/datasetV6/NI',
                   'F:/insectsDCT_datasets/classifier/datasetV6/Orchard',
                   'F:/insectsDCT_datasets/classifier/datasetV6/NI2_MAMBO',
                   'F:/insectsDCT_datasets/classifier/datasetV6/GBIF'
                  ]

# Classification dataset V7
image_path_list = ['F:/insectsDCT_datasets/classifier/datasetV6/NI2',
                   'F:/insectsDCT_datasets/classifier/datasetV6/NI',
                   'F:/insectsDCT_datasets/classifier/datasetV6/Orchard',
                   'F:/insectsDCT_datasets/classifier/datasetV6/NI2_MAMBO',
                   'F:/insectsDCT_datasets/classifier/datasetV6/GBIF',
                   'F:/insectsDCT_datasets/classifier/datasetV7/LepiOtar',
                   'F:/insectsDCT_datasets/classifier/datasetV7/ACSHQ'
                  ]

datasetLoader = HierarchicalDatasetLoader(image_path_list, 5)
path_names = datasetLoader.get_path_names()
hierarchyL1, hierarchyL2, labelsL1, labelsL2, labelsL3 = datasetLoader.get_hierarchy_labels()

level1 = {}
for l, t, v in zip(labelsL1, datasetLoader.get_cls_num_list(0), datasetLoader.get_cls_num_list(0, 1)):
    level1[l] = [t, v]
    
level2 = {}
for l, t, v in zip(labelsL2, datasetLoader.get_cls_num_list(1), datasetLoader.get_cls_num_list(1, 1)):
    level2[l] = [t, v]
    
level3 = {}
for l, t, v in zip(labelsL3, datasetLoader.get_cls_num_list(2), datasetLoader.get_cls_num_list(2, 1)):
    level3[l] = [t, v]
        
print(len(level1))
print(len(level2))
print(len(level3))

fig, ax = plt.subplots(3, sharex=True)
fig.tight_layout(pad=2.0)
level1values = sorted([sum(val) for val in list(level1.values())], reverse=True)
print(sum(level1values))
ax[0].plot(level1values, 'bo-', linewidth=1)
ax[0].set_xlim([0, 110])
ax[0].set_title("Level 1")
ax[0].set_ylabel("images")
level2values = sorted([sum(val) for val in list(level2.values())], reverse=True)
print(sum(level2values))
ax[1].plot(level2values, 'ro-', linewidth=1)
ax[1].set_xlim([0, 110])
ax[1].set_title("Level 2")
ax[1].set_ylabel("images")
level3values = sorted([sum(val) for val in list(level3.values())], reverse=True)
print(sum(level3values))
ax[2].plot(level3values, 'go-', linewidth=1)
ax[2].set_xlim([0, 110])
ax[2].set_title("Level 3")
ax[2].set_ylabel("images")
ax[2].set_xlabel("class index")
plt.show()


