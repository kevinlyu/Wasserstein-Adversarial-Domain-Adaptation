import torch
import torch.nn as nn
from sklearn.manifold import TSNE
import os
import numpy as np
from utils import *
from dataloaders import *
from models import *


class DANN:
    def __init__(self, components, optimizers, dataloaderes, criterions, total_epoch, feature_dim, class_num, log_interval):

        self.extractor = components["extractor"]
        self.classifier = components["classifier"]
        self.discriminator = components["discriminator"]

        self.class_criterion = criterions["class"]
        self.domain_criterion = criterions["domain"]

        self.opt = optimizers["opt"]

        self.src_loader = dataloaderes["source_loader"]
        self.tar_loader = dataloaderes["target_loader"]
        self.test_src_loader = dataloaderes["test_src_loader"]
        self.test_tar_loader = dataloaderes["test_tar_loader"]

        self.total_epoch = total_epoch
        self.log_interval = log_interval
        self.feature_dim = feature_dim
        self.class_num = class_num
        self.img_size = 28

    def train(self):
        print("[Training]")

        for epoch in range(self.total_epoch):

            start_steps = epoch*len(self.src_loader)
            total_steps = self.total_epoch*len(self.tar_loader)

            for index, (src, tar) in enumerate(zip(self.src_loader, self.tar_loader)):

                p = float(index + start_steps)/total_steps
                constant = 2.0 / (1.0+np.exp(-10*p))-1

                src_data, src_label = src
                tar_data, tar_label = tar

                size = min(src_data.shape[0], tar_data.shape[0])
                src_data, src_label = src_data[0:size], src_label[0:size]
                tar_data, tar_label = tar_data[0:size], tar_label[0:size]

                """ For MNIST data, expand number of channel to 3 """
                if src_data.shape[1] != 3:
                    src_data = src_data.expand(
                        src_data.shape[0], 3, self.img_size, self.img_size)

                src_data, src_label = src_data.cuda(), src_label.cuda()
                tar_data, tar_label = tar_data.cuda(), tar_label.cuda()

                """ train classifer """

                self.opt.zero_grad()

                src_z = self.extractor(src_data)
                tar_z = self.extractor(tar_data)

                pred_class = self.classifier(src_z)
                class_loss = self.class_criterion(pred_class, src_label)

                ''' classify accuracy '''
                _, predicted = torch.max(pred_class, 1)

                accuracy = 100.0 * \
                    (predicted == src_label).sum() / src_data.size(0)

                pred_d_src = self.discriminator(src_z, p)
                pred_d_tar = self.discriminator(tar_z, p)

                d_loss_src = self.domain_criterion(pred_d_src, torch.zeros(
                    src_z.size(0)).type(torch.LongTensor).cuda())
                d_loss_tar = self.domain_criterion(pred_d_tar, torch.ones(
                    tar_z.size(0)).type(torch.LongTensor).cuda())

                domain_loss = d_loss_src + d_loss_tar

                loss = class_loss + domain_loss
                loss.backward()
                self.opt.step()

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

    def save_model(self, path="./saved_DANN/"):
        try:
            os.stat(path)
        except:
            os.mkdir(path)

        torch.save(self.extractor.state_dict(),
                   os.path.join(path, "DANN_E.pkl"))
        torch.save(self.classifier.state_dict(),
                   os.path.join(path, "DANN_C.pkl"))
        torch.save(self.discriminator.state_dict(),
                   os.path.join(path, "DANN_D.pkl"))

    def load_model(self, path="./saved_DANN/"):

        self.extractor.load_state_dict(
            torch.load(os.path.join(path, "DANN_E.pkl")))
        self.classifier.load_state_dict(
            torch.load(os.path.join(path, "DANN_C.pkl")))
        self.discriminator.load_state_dict(
            torch.load(os.path.join(path, "DANN_D.pkl")))

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
        # modefied perplexity
        tsne = TSNE(n_components=dim, n_iter=3000, perplexity=35)

        embedding = tsne.fit_transform(data)
        embedding_max, embedding_min = np.max(
            embedding, 0), np.min(embedding, 0)
        embedding = (embedding-embedding_min) / (embedding_max - embedding_min)

        if dim == 2:
            visualize_2d("./saved_DANN/", embedding,
                         label, tag, self.class_num)

        elif dim == 3:
            visualize_3d("./saved_DANN/", embedding,
                         label, tag, self.class_num)


if __name__ == "__main__":

    batch_size = 50
    total_epoch = 300
    # total_epoch = 250
    feature_dim = 1000
    class_num = 10
    log_interval = 10
    partial = True

    source_loader = torch.utils.data.DataLoader(datasets.MNIST(
        "../dataset/mnist/", train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])), batch_size=batch_size, shuffle=True)

    target_loader = torch.utils.data.DataLoader(MNISTM(
        transform=transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]), train=True, partial=partial), batch_size=batch_size, shuffle=True)

    test_src_loader = torch.utils.data.DataLoader(datasets.MNIST(
        "../dataset/mnist/", train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])), batch_size=batch_size, shuffle=True)

    test_tar_loader = torch.utils.data.DataLoader(MNISTM(
        transform=transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]), train=False, partial=partial),  batch_size=batch_size, shuffle=True)

    extractor = Extractor_new(encoded_dim=feature_dim).cuda()
    classifier = Classifier(encoded_dim=feature_dim,
                            class_num=class_num).cuda()
    discriminator = Discriminator_GRL(encoded_dim=feature_dim).cuda()

    class_criterion = nn.NLLLoss()
    domain_criterion = nn.NLLLoss()

    opt = torch.optim.Adam([{"params": classifier.parameters()},
                            {"params": extractor.parameters()},
                            {"params": discriminator.parameters()}], lr=1e-3)
    # baseline 5e-4

    components = {"extractor": extractor,
                  "classifier": classifier, "discriminator": discriminator}

    optimizers = {"opt": opt}

    dataloaders = {"source_loader": source_loader, "target_loader": target_loader,
                   "test_src_loader": test_src_loader, "test_tar_loader": test_tar_loader}

    criterions = {"class": class_criterion, "domain": domain_criterion}

    model = DANN(components, optimizers, dataloaders,
                 criterions, total_epoch, feature_dim, class_num, log_interval)

    model.train()
    #model.save_model()
    #model.load_model()
    model.test()
    model.visualize(dim=2)
    # model.visualize(dim=3)
