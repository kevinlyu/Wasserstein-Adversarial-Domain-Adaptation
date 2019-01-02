import torch
import torch.nn as nn
from sklearn.manifold import TSNE
import os
import numpy as np
from utils import *
from dataloaders import *
from models import *

from PIL import Image
import matplotlib.pyplot as plt


class WDGRL:

    def __init__(self, components, optimizers, dataloaders, criterions, total_epoch, feature_dim, class_num, log_interval):
        self.extractor = components["extractor"]
        self.classifier = components["classifier"]
        self.discriminator = components["discriminator"]

        self.class_criterion = criterions["class"]

        self.c_opt = optimizers["c_opt"]
        self.d_opt = optimizers["d_opt"]

        self.src_loader = dataloaders["source_loader"]
        self.tar_loader = dataloaders["target_loader"]
        self.test_src_loader = dataloaders["test_src_loader"]
        self.test_tar_loader = dataloaders["test_tar_loader"]

        self.total_epoch = total_epoch
        self.log_interval = log_interval
        self.feature_dim = feature_dim
        self.class_num = class_num
        self.img_size = 28

    def train(self):
        print("[Training]")

        for epoch in range(self.total_epoch):

            for index, (src, tar) in enumerate(zip(self.src_loader, self.tar_loader)):

                src_data, src_label = src
                tar_data, tar_label = tar

                size = min(src_data.shape[0], tar_data.shape[0])
                src_data, src_label = src_data[0:size], src_label[0:size]
                tar_data, tar_label = tar_data[0:size], tar_label[0:size]

                """ For MNIST """
                if src_data.shape[1] != 3:
                    src_data = src_data.expand(
                        src_data.shape[0], 3, self.img_size, self.img_size)

                src_data, src_label = src_data.cuda(), src_label.cuda()
                tar_data, tar_label = tar_data.cuda(), tar_label.cuda()

                """ train classifer """

                set_requires_grad(self.extractor, requires_grad=True)
                set_requires_grad(self.discriminator, requires_grad=False)

                src_z = self.extractor(src_data)
                tar_z = self.extractor(tar_data)

                pred_class = self.classifier(src_z)
                class_loss = self.class_criterion(pred_class, src_label)

                wasserstein_diatance = self.discriminator(
                    src_z).mean() - self.discriminator(tar_z).mean()

                loss = class_loss + wasserstein_diatance
                c_opt.zero_grad()
                loss.backward()
                c_opt.step()

                """ classify accuracy """
                _, predicted = torch.max(pred_class, 1)
                accuracy = 100.0 * \
                    (predicted == src_label).sum() / src_data.size(0)

                """ train discriminator """

                set_requires_grad(self.extractor, requires_grad=False)
                set_requires_grad(self.discriminator, requires_grad=True)

                with torch.no_grad():
                    src_z = self.extractor(src_data)
                    tar_z = self.extractor(tar_data)

                for _ in range(5):
                    gp = gradient_penalty(self.discriminator, src_z, tar_z)
                    d_src_loss = self.discriminator(src_z)
                    d_tar_loss = self.discriminator(tar_z)

                    wasserstein_distance = d_src_loss.mean()-d_tar_loss.mean()

                    domain_loss = -wasserstein_distance + 10*gp

                    d_opt.zero_grad()
                    domain_loss.backward()
                    d_opt.step()

                loss = loss+domain_loss

                if index % self.log_interval == 0:
                    print("[Epoch {:3d}] Total_loss: {:.4f}   C_loss: {:.4f}   D_loss:{:.4f}".format(
                        epoch, loss, class_loss, domain_loss))
                    print("Classifier Accuracy: {:.2f}\n".format(accuracy))

    def test(self):
        print("[Testing]")

        self.extractor.cuda().eval()
        self.classifier.cuda().eval()

        src_correct = 0
        tar_correct = 0

        # testing source
        for index, src in enumerate(self.test_src_loader):
            src_data, src_label = src
            src_data, src_label = src_data.cuda(), src_label.cuda()

            ''' for MNIST '''
            if src_data.shape[1] != 3:
                src_data = src_data.expand(
                    src_data.shape[0], 3, self.img_size, self.img_size)

            src_z = self.extractor(src_data)
            src_pred = self.classifier(src_z)
            _, predicted = torch.max(src_pred, 1)
            src_correct += (predicted == src_label).sum().item()

        # testing target
        for index, (src, tar) in enumerate(zip(self.test_src_loader, self.test_tar_loader)):

            tar_data, tar_label = tar
            tar_data, tar_label = tar_data.cuda(), tar_label.cuda()

            tar_z = self.extractor(tar_data)
            tar_pred = self.classifier(tar_z)
            _, predicted = torch.max(tar_pred, 1)
            tar_correct += (predicted == tar_label).sum().item()

        print("source accuracy: {:.2f}%".format(
            100*src_correct/len(self.test_src_loader.dataset)))
        print("target accuracy: {:.2f}%".format(
            100*tar_correct/len(self.test_tar_loader.dataset)))

    def save_model(self, path="./saved_WDGRL/"):
        try:
            os.stat(path)
        except:
            os.mkdir(path)

        torch.save(self.extractor.state_dict(),
                   os.path.join(path, "WDGRL_E.pkl"))
        torch.save(self.classifier.state_dict(),
                   os.path.join(path, "WDGRL_C.pkl"))
        torch.save(self.discriminator.state_dict(),
                   os.path.join(path, "WDGRL_D.pkl"))

    def load_model(self, path="./saved_WDGRL/"):

        self.extractor.load_state_dict(
            torch.load(os.path.join(path, "WDGRL_E.pkl")))
        self.classifier.load_state_dict(
            torch.load(os.path.join(path, "WDGRL_C.pkl")))
        self.discriminator.load_state_dict(
            torch.load(os.path.join(path, "WDGRL_D.pkl")))

    def visualize(self, dim=2, plot_num=1000):
        print("t-SNE reduces to dimension {}".format(dim))

        self.extractor.cpu().eval()

        src_data = torch.FloatTensor()
        tar_data = torch.FloatTensor()

        ''' If use USPS dataset, change it to IntTensor() '''
        src_label = torch.LongTensor()
        tar_label = torch.LongTensor()

        for index, src in enumerate(self.src_loader):
            data, label = src
            src_data = torch.cat((src_data, data))
            src_label = torch.cat((src_label, label))

        for index, tar in enumerate(self.tar_loader):
            data, label = tar
            tar_data = torch.cat((tar_data, data))
            tar_label = torch.cat((tar_label, label))

        ''' for MNIST dataset '''
        if src_data.shape[1] != 3:
            src_data = src_data.expand(
                src_data.shape[0], 3, self.img_size, self.img_size)

        src_data, src_label = src_data[0:plot_num], src_label[0:plot_num]
        tar_data, tar_label = tar_data[0:plot_num], tar_label[0:plot_num]

        src_z = self.extractor(src_data)
        tar_z = self.extractor(tar_data)

        data = np.concatenate((src_z.detach().numpy(), tar_z.detach().numpy()))
        label = np.concatenate((src_label.numpy(), tar_label.numpy()))

        src_tag = torch.zeros(src_z.size(0))
        tar_tag = torch.ones(tar_z.size(0))
        tag = np.concatenate((src_tag.numpy(), tar_tag.numpy()))

        ''' t-SNE process '''
        tsne = TSNE(n_components=dim)

        embedding = tsne.fit_transform(data)

        embedding_max, embedding_min = np.max(
            embedding, 0), np.min(embedding, 0)
        embedding = (embedding-embedding_min) / (embedding_max - embedding_min)

        if dim == 2:
            visualize_2d("./saved_WDGRL/", embedding,
                         label, tag, self.class_num)

        elif dim == 3:
            visualize_3d("./saved_WDGRL/", embedding,
                         label, tag, self.class_num)


''' Unit test '''
if __name__ == "__main__":
    print("WDGRL model")

    batch_size = 150
    total_epoch = 200
    feature_dim = 1000
    class_num = 10
    log_interval = 10

    source_loader = torch.utils.data.DataLoader(datasets.MNIST(
        "../dataset/mnist/", train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])), batch_size=batch_size, shuffle=True)

    target_loader = torch.utils.data.DataLoader(USPS(
        transform=transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]), train=True, partial=True), batch_size=batch_size, shuffle=True)

    test_src_loader = torch.utils.data.DataLoader(datasets.MNIST(
        "../dataset/mnist/", train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])), batch_size=batch_size, shuffle=True)

    test_tar_loader = torch.utils.data.DataLoader(USPS(
        transform=transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]), train=False, partial=True),  batch_size=batch_size, shuffle=True)

    extractor = Extractor_new(encoded_dim=feature_dim).cuda()
    classifier = Classifier(encoded_dim=feature_dim,
                            class_num=class_num).cuda()
    discriminator = Discriminator_WGAN(encoded_dim=feature_dim).cuda()

    class_criterion = nn.CrossEntropyLoss()

    c_opt = torch.optim.RMSprop([{"params": classifier.parameters()},
                                 {"params": extractor.parameters()}], lr=1e-3)
    d_opt = torch.optim.RMSprop(discriminator.parameters(), lr=1e-3)

    components = {"extractor": extractor,
                  "classifier": classifier, "discriminator": discriminator}
    optimizers = {"c_opt": c_opt, "d_opt": d_opt}
    dataloaders = {"source_loader": source_loader, "target_loader": target_loader,
                   "test_src_loader": test_src_loader, "test_tar_loader": test_tar_loader}

    criterions = {"class": class_criterion}

    model = WDGRL(components, optimizers, dataloaders, criterions,
                  total_epoch, feature_dim, class_num, log_interval)
    # model.load_model()
    model.train()
    model.save_model()
    model.load_model()
    model.test()
    model.visualize(dim=2)
    # model.visualize(dim=3)
    # model.load_model()
