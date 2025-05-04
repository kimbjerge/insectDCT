'''Configurations
'''

import argparse

parser = argparse.ArgumentParser()

#parser.add_argument('--train_path', type=str, help='Specify the path to the train images.', default='O:/Tech_TTH-KBE/NI_2/Datasets/trainNI')
#parser.add_argument('--test_path', type=str, help='Specify the path to the validate images.', default='O:/Tech_TTH-KBE/NI_2/Datasets/valNI')
parser.add_argument('--testNI_path', type=str, help='Specify the path to the test images.', default='O:/Tech_TTH-KBE/NI_2/Datasets/testNINew')
#parser.add_argument('--testNI_path', type=str, help='Specify the path to the test images.', default='/mnt/Dfs/Tech_TTH-KBE/NI_2/Datasets/testNI')
parser.add_argument('--testNI2_path', type=str, help='Specify the path to the test images.', default='/mnt/Dfs/Tech_TTH-KBE/NI_2/Datasets/classesNI2')
#parser.add_argument('--testNI_path', type=str, help='Specify the path to the test images.', default='/mnt/Dfs/Tech_TTH-KBE/NI_2/Datasets/testNINew')
parser.add_argument('--train_path', type=str, help='Specify the path to the train csv file.', default='/home/don/data/Arthropods/NItrain')
parser.add_argument('--test_path', type=str, help='Specify the path to the test csv file.', default='/home/don/data/Arthropods/NIval')
parser.add_argument('--detect_path', type=str, help='Specify the path to the test csv file.', default='O:/Tech_Insects/Datasets/insects/testOtherSorted/test1201otherSquared')
#parser.add_argument('--detect_path', type=str, help='Specify the path to the test images.', default='O:/Tech_TTH-KBE/NI_2/Datasets/TestNINew/Hymenoptera Apidae Bombus')
#parser.add_argument('--detect_path', type=str, help='Specify the path to the test images.', default='O:/Tech_TTH-KBE/NI_2/Datasets/TestNINew/Diptera Syrhidae')
#parser.add_argument('--detect_path', type=str, help='Specify the path to the test images.', default='O:/Tech_TTH-KBE/NI_2/Datasets/TestNINew/Hymenoptera Apidae')
#parser.add_argument('--detect_path', type=str, help='Specify the path to the test images.', default='O:/Tech_TTH-KBE/NI_2/Datasets/TestNINew/Diptera Muscidae')

#parser.add_argument('--weights', type=str, help='Specify the path to save the model.', default='dhc0_20_best.pth')
#parser.add_argument('--weights', type=str, help='Specify the path to save the model.', default='dhctfv2-128-e13.pth')
parser.add_argument('--weights', type=str, help='Specify the path to save the model.', default='dhc_best.pth')
#parser.add_argument('--weights', type=str, help='Specify the path to save the model.', default='dhc_save1_00_best.pth')
#parser.add_argument('--weights', type=str, help='Specify the path to save the model.', default='dhcv2-128-e25.pth')

parser.add_argument('--model_save_path', type=str, help='Specify the path to save the model.', default='./saved/')
parser.add_argument('--graphs_folder', type=str, help='Specify the path to save the graphs.', default='./graph_folder/')
parser.add_argument('--epoch', type=int, help='Specify the number of epochs for the training.', default=20) # 30
#parser.add_argument('--batch_size', type=int, help='Specify the batch size to be used during training/testing.', default=10)
parser.add_argument('--batch_size', type=int, help='Specify the batch size to be used during training/testing.', default=20)
parser.add_argument('--optimizer', type=str, help='Specify the optimizer to be used during training. (Adam or SGD)', default='Adam')
parser.add_argument('--learning_rate', type=float, help='Specify the learning rate (SGD or Adam) to be used during training.', default=1e-4)
#parser.add_argument('--optimizer', type=str, help='Specify the optimizer to be used during training. (Adam or SGD)', default='SGD')
#parser.add_argument('--learning_rate', type=float, help='Specify the learning rate (SGD or Adam) to be used during training.', default=1e-3)
parser.add_argument('--momentum', type=float, help='Specify the SGD momentum to be used during training.', default=1e-4)
parser.add_argument('--weight_decay', type=float, help='Specify the SGD weitht_decay to be used during training.', default=1e-4)
parser.add_argument('--num_workers', type=int, help='Specify the number of workers to be used to load the data.', default=0)
parser.add_argument('--img_size', type=int, help='Specify the size of the input image.', default=128)
parser.add_argument('--img_depth', type=int, help='Specify the depth of the input image.', default=3)
parser.add_argument('--device', type=str, help='Specify which device to be used for the evaluation. Either "cpu" or "gpu".', default='gpu')

args = parser.parse_args()
