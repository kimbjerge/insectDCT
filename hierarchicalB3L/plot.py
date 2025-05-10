'''
Graph plotting functions.
'''

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

fig = plt.figure(figsize=(20, 5))

def plot_loss_acc(path, num_epoch, train_accuracies_level1, train_accuracies_level2, train_accuracies_level3, train_losses,
                    test_accuracies_level1, test_accuracies_level2, test_accuracies_level3, test_losses):
    '''
    Plot line graphs for the accuracies and loss at every epochs for both training and testing.
    '''
    
    #plt.clf()

    epochs = [x+1 for x in range(num_epoch+1)]
 
    train_level1_accuracy_df = pd.DataFrame({"Epochs":epochs, "Accuracy":train_accuracies_level1, "Level":['L1']*(num_epoch+1), "Mode":"train"})
    train_level2_accuracy_df = pd.DataFrame({"Epochs":epochs, "Accuracy":train_accuracies_level2, "Level":['L2']*(num_epoch+1), "Mode":"train"})
    train_level3_accuracy_df = pd.DataFrame({"Epochs":epochs, "Accuracy":train_accuracies_level3, "Level":['L3']*(num_epoch+1), "Mode":"train"})
    test_level1_accuracy_df = pd.DataFrame({"Epochs":epochs, "Accuracy":test_accuracies_level1, "Level":['L1']*(num_epoch+1), "Mode":"validate"})
    test_level2_accuracy_df = pd.DataFrame({"Epochs":epochs, "Accuracy":test_accuracies_level2, "Level":['L2']*(num_epoch+1), "Mode":"validate"})
    test_level3_accuracy_df = pd.DataFrame({"Epochs":epochs, "Accuracy":test_accuracies_level3, "Level":['L3']*(num_epoch+1), "Mode":"validate"})

    data_level1 = pd.concat([train_level1_accuracy_df, test_level1_accuracy_df])
    data_level2 = pd.concat([train_level2_accuracy_df, test_level2_accuracy_df])
    data_level3 = pd.concat([train_level3_accuracy_df, test_level3_accuracy_df])

    data_levels = pd.concat([train_level1_accuracy_df, test_level1_accuracy_df, 
                             train_level2_accuracy_df, test_level2_accuracy_df, 
                             train_level3_accuracy_df, test_level3_accuracy_df])

    plt.show()
    
    sns.lineplot(data=data_levels, x='Epochs', y='Accuracy', hue='Level', style='Mode')
    plt.xticks(np.arange(1, 20, 2))
    plt.title('Train and validation accuracy')
    plt.savefig(path+f'accuracy_train_validation_epoch.png')

    plt.show()

    sns.lineplot(data=data_level1, x='Epochs', y='Accuracy', style='Mode')
    plt.title('L1 Family Accuracy')
    plt.savefig(path+f'accuracy_level1_epoch.png')

    plt.show()
    
    sns.lineplot(data=data_level2, x='Epochs', y='Accuracy', style='Mode')
    plt.title('L2 Order Accuracy')
    plt.savefig(path+f'accuracy_level2_epoch.png')

    plt.show()
    
    sns.lineplot(data=data_level3, x='Epochs', y='Accuracy', style='Mode')
    plt.title('L3 Species Accuracy')
    plt.savefig(path+f'accuracy_level3_epoch.png')

    plt.show()

    train_loss_df = pd.DataFrame({"Epochs":epochs, "Loss":train_losses, "Mode":['train']*(num_epoch+1)})
    test_loss_df = pd.DataFrame({"Epochs":epochs, "Loss":test_losses, "Mode":['validate']*(num_epoch+1)})

    data = pd.concat([train_loss_df, test_loss_df])

    sns.lineplot(data=data, x='Epochs', y='Loss', style='Mode')
    plt.xticks(np.arange(1, 20, 2))
    plt.title('Loss')

    plt.savefig(path+f'loss_epoch.png')
    plt.show()

    return None
