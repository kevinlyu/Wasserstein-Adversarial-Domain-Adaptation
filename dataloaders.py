import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms, utils


class MNISTM(Dataset):
    '''
    Definition of MNISTM dataset
    '''

    def __init__(self, root="/home/neo/dataset/mnistm/", train=True, partial=False, transform=None, target_transform=None):
        super(MNISTM, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.partial = partial

        if self.train:
            if not self.partial:
                self.data, self.label = torch.load(
                    os.path.join(self.root, "mnistm_pytorch_train"))
            else:
                self.data, self.label = torch.load(
                    os.path.join(self.root, "partial_mnistm_pytorch_train"))
        else:
            if not self.partial:
                self.data, self.label = torch.load(
                    os.path.join(self.root, "mnistm_pytorch_test"))
            else:
                self.data, self.label = torch.load(
                    os.path.join(self.root, "partial_mnistm_pytorch_test"))

    def __getitem__(self, index):

        data, label = self.data[index], self.label[index]
        data = Image.fromarray(data.squeeze().numpy(), mode="RGB")

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return data, label

    def __len__(self):
        # Return size of dataset
        return len(self.data)


class USPS(Dataset):
    '''
    Definition of USPS dataset
    '''

    def __init__(self, root="/home/neo/dataset/usps/", train=True, partial=False, transform=None, target_transform=None):
        super(USPS, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.partial = partial

        if self.train:
            if not self.partial:
                self.data, self.label = torch.load(
                    os.path.join(self.root, "usps_pytorch_train"))
            else:
                self.data, self.label = torch.load(
                    os.path.join(self.root, "partial_usps_pytorch_train"))
        else:
            if not self.partial:
                self.data, self.label = torch.load(
                    os.path.join(self.root, "usps_pytorch_test"))
            else:
                self.data, self.label = torch.load(
                    os.path.join(self.root, "partial_usps_pytorch_test"))

    def __getitem__(self, index):
        data, label = self.data[index], self.label[index]
        data = np.stack([data]*3, axis=2)
        data = data.astype(float)
        data = Image.fromarray(np.uint8(data)*255, mode="RGB")

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return data, label

    def __len__(self):
        return len(self.data)


########################################################################
# VisDA 2018
visda_syn_root = "../dataset/visda2018/train/"
VisdaSyn = torchvision.datasets.ImageFolder(visda_syn_root, transform=transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))

visda_syn_loader = torch.utils.data.DataLoader(
    VisdaSyn, batch_size=100, shuffle=True, num_workers=4)

visda_real_root = "../dataset/visda2018/validation/"

VisdaReal = torchvision.datasets.ImageFolder(visda_real_root, transform=transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))

visda_real_loader = torch.utils.data.DataLoader(
    VisdaReal, batch_size=100, shuffle=True, num_workers=4)


########################################################################
def get_office_loader(domain, partial, batch_size=50):

    if domain == "amazon":
        loader = torch.utils.data.DataLoader(Amazon(
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
            ]), partial=partial), batch_size=batch_size, shuffle=True)

    elif domain == "webcam":
        loader = torch.utils.data.DataLoader(Webcam(
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
            ]), partial=partial), batch_size=batch_size, shuffle=True)

    elif domain == "dslr":
        loader = torch.utils.data.DataLoader(DSLR(
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
            ]), partial=partial), batch_size=batch_size, shuffle=True)

    return loader


class Amazon(Dataset):

    def __init__(self, root="../dataset/office31/", train=True, partial=False, transform=None, target_transform=None):
        super(Amazon, self).__init__()
        self.root = root
        self.train = train
        self.partial = partial
        self.transform = transform
        self.target_transform = target_transform

        if self.partial:
            amazon = np.load(os.path.join(self.root, "amazon10.npz"))
        else:
            amazon = np.load(os.path.join(self.root, "amazon31.npz"))

        self.data, self.label = amazon["data"], amazon["label"]

    def __getitem__(self, index):

        data, label = self.data[index], self.label[index]
        data = Image.fromarray(np.uint8(data*255.0), mode="RGB")

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return data, label

    def __len__(self):
        return (len(self.label))


class Webcam(Dataset):

    def __init__(self, root="../dataset/office31/", train=True, partial=False, transform=None, target_transform=None):
        super(Webcam, self).__init__()
        self.root = root
        self.train = train
        self.partial = partial
        self.transform = transform
        self.target_transform = target_transform

        if self.partial:
            webcam = np.load(os.path.join(self.root, "webcam10.npz"))
        else:
            webcam = np.load(os.path.join(self.root, "webcam31.npz"))

        self.data, self.label = webcam["data"], webcam["label"]

    def __getitem__(self, index):

        data, label = self.data[index], self.label[index]
        data = Image.fromarray(np.uint8(data*255.0), mode="RGB")

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return data, label

    def __len__(self):
        return (len(self.label))


class DSLR(Dataset):

    def __init__(self, root="../dataset/office31/", train=True, partial=False, transform=None, target_transform=None):
        super(DSLR, self).__init__()
        self.root = root
        self.train = train
        self.partial = partial
        self.transform = transform
        self.target_transform = target_transform

        if self.partial:
            dslr = np.load(os.path.join(self.root, "dslr10.npz"))
        else:
            dslr = np.load(os.path.join(self.root, "dslr31.npz"))

        self.data, self.label = dslr["data"], dslr["label"]

        """
        id = np.random.randint(len(self.label))
        plt.imshow(self.data[id])
        plt.show()
        exit()
        """

    def __getitem__(self, index):

        data, label = self.data[index], self.label[index]
        data = Image.fromarray(np.uint8(data*255.0), mode="RGB")

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return data, label

    def __len__(self):
        return (len(self.label))


if __name__ == "__main__":

    f = get_office_loader("webcam", partial=False)
    for idx, data in enumerate(f):
        img, label = data

        fig = plt.figure()
        grid = utils.make_grid(img)
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
        plt.show()
        exit()
