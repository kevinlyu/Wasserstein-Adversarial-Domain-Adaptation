from models import *
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


def partial_mnistm(mnistm_path="/home/neo/dataset/mnistm/"):
    train_data, train_label = torch.load(
        os.path.join(mnistm_path, "mnistm_pytorch_train"))
    test_data, test_label = torch.load(
        os.path.join(mnistm_path, "mnistm_pytorch_test"))

    train_data = train_data.numpy()
    train_label = train_label.numpy()
    test_data = test_data.numpy()
    test_label = test_label.numpy()

    """ Training Data """
    idx = np.where(train_label % 3 == 0)
    train_data = train_data[idx]
    train_label = train_label[idx]

    train_data = torch.from_numpy(train_data)
    train_label = torch.from_numpy(train_label)

    """ Testing Data """
    idx = np.where(test_label % 3 == 0)
    test_data = test_data[idx]
    test_label = test_label[idx]

    test_data = torch.from_numpy(test_data)
    test_label = torch.from_numpy(test_label)

    """ Save Dataset """
    partial_train = (train_data, train_label)
    partial_test = (test_data, test_label)

    with open(os.path.join(mnistm_path, "partial_mnistm_pytorch_train"), "wb") as f:
        torch.save(partial_train, f)

    with open(os.path.join(mnistm_path, "partial_mnistm_pytorch_test"), "wb") as f:
        torch.save(partial_test, f)

    print("Done")


def partial_usps(usps_path="/home/neo/dataset/usps/"):

    train_data, train_label = torch.load(
        os.path.join(usps_path, "usps_pytorch_train"))
    test_data, test_label = torch.load(
        os.path.join(usps_path, "usps_pytorch_test"))
    
    """ Training Data """
    idx = np.where(train_label % 3 == 0)
    train_data = train_data[idx]
    train_label = train_label[idx]

    """ Testing Data """
    idx = np.where(test_label % 3 == 0)
    test_data = test_data[idx]
    test_label = test_label[idx]

    """ Save Dataset """
    partial_train = (train_data, train_label)
    partial_test = (test_data, test_label)

    with open(os.path.join(usps_path, "partial_usps_pytorch_train"), "wb") as f:
        torch.save(partial_train, f)

    with open(os.path.join(usps_path, "partial_usps_pytorch_test"), "wb") as f:
        torch.save(partial_test, f)
    
    print("Done")

partial_usps()
partial_mnistm()
