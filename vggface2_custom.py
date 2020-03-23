
import os
from glob import glob

from PIL import Image
import numpy as np


def vgg_load(train=True, select_classes=150, start_ind=0):

    data_x = []
    data_y = []
    
    start_label = start_ind
    end_label = start_label + select_classes

    if train:
        data_path = 'd:/datasets/vggface2_custom/trainset'
    else:
        data_path = 'd:/datasets/vggface2_custom/validset'
    label_list = sorted(os.listdir(data_path))
    selected_label_list = label_list[start_label:end_label]


    label_vocab = dict()
    for i in range(len(selected_label_list)):
        label_vocab[selected_label_list[i]] = i


    for label in selected_label_list:

        file_list = os.listdir(os.path.join(data_path, label))

        for i in range(len(file_list)):
            file_path = os.path.join(data_path, label, file_list[i])
            image = np.array(Image.open(file_path))
            data_x.append(image)
            data_y.append(label_vocab[label])        

    data_x = np.array(data_x)
    data_y = np.array(data_y)    

    return (data_x, data_y)


if __name__ == '__main__':
    (train_x, train_y) = vgg_load(train=True)
    (valid_x, valid_y) = vgg_load(train=False)
    print('train_x shape : ', train_x.shape)
    print('train_y shape : ', train_y.shape)
    print('valid_x shape : ', valid_x.shape)
    print('valid_y shape : ', valid_y.shape)


    



