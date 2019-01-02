import os
import pickle

import h5py
import numpy as np
import torch
# from torch.utils.data import Dataset
from torchvision import datasets
from PIL import Image

import matplotlib.pyplot as plt
'''
image after preprocess is in CHW (channel, height, width) format for pytorch use. If you want to plot 
these images, convert them into HWC first
'''


def process_mnistm(mnistm_path="/home/neo/dataset/mnistm/", mnist_path="/home/neo/dataset/mnist/"):
    '''
    reference: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/pixelda/mnistm.py
    mnistm pickle file: https://github.com/VanushVaswani/keras_mnistm/releases/download/1.0/keras_mnistm.pkl.gz
    '''
    with open(os.path.join(mnistm_path, "mnistm.pkl"), "rb") as f:
        mnistm_data = pickle.load(f, encoding="bytes")

    # load content of pickle file into pytorch tensor (data only)
    mnistm_train_data = torch.ByteTensor(mnistm_data[b'train'])
    mnistm_test_data = torch.ByteTensor(mnistm_data[b'test'])

    # load label of mnist dataset
    mnist_train_labels = datasets.MNIST(
        root=mnist_path, train=True, download=True).train_labels
    mnist_test_labels = datasets.MNIST(
        root=mnist_path, train=False, download=True).test_labels

    # combine (data, label)
    training_set = (mnistm_train_data, mnist_train_labels)
    testing_set = (mnistm_test_data, mnist_test_labels)

    # save mnist data with (data, label)
    with open(os.path.join(mnistm_path, "mnistm_pytorch_train"), "wb") as f:
        torch.save(training_set, f)

    with open(os.path.join(mnistm_path, "mnistm_pytorch_test"), "wb") as f:
        torch.save(testing_set, f)

    print("Done!")


def process_usps(usps_path="/home/neo/dataset/usps/"):
    '''
    download .h5 format usps dataset from Kaggle site 
    '''
    with h5py.File(os.path.join(usps_path, "usps.h5"), "r") as f:
        train = f.get("train")
        usps_train_data = train.get("data")[:]
        usps_train_label = train.get("target")[:]
        test = f.get("test")
        usps_test_data = test.get("data")[:]
        usps_test_label = test.get("target")[:]

    #usps_train_data = np.asarray(usps_train_data)
    #usps_test_data = np.asarray(usps_test_data)

    usps_train_data = np.reshape(usps_train_data, (-1, 16, 16))
    usps_test_data = np.reshape(usps_test_data, (-1, 16, 16))

    usps_train_data *= 255.0
    usps_test_data *= 255.0

    usps_train_label = usps_train_label.astype(int)
    usps_test_label = usps_test_label.astype(int)

    training_set = (usps_train_data, usps_train_label)
    testing_set = (usps_test_data, usps_test_label)

    # save usps data with (data, label)
    with open(os.path.join(usps_path, "usps_pytorch_train"), "wb") as f:
        torch.save(training_set, f)

    with open(os.path.join(usps_path, "usps_pytorch_test"), "wb") as f:
        torch.save(testing_set, f)

    print("Done")

# process_mnistm()
# process_usps()
