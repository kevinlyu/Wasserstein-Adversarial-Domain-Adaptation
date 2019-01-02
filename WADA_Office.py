import torch
import torch.nn as nn
from sklearn.manifold import TSNE
import os
import numpy as np
from utils import *
from dataloaders import *
from models_office import *

from PIL import Image
import matplotlib.pyplot as plt

from torchvision import transforms, utils


class WADA_II:

    def __init__(self, components, optimizers, dataloaders, criterions, total_epoch, class_num, log_interval):
        self.src_extractor = components["src_extractor"]
        self.tar_extractor = components["tar_extractor"]
        self.classifier = components["classifier"]
        self.discriminator = components["discriminator"]
        self.relater = components["relater"]

        self.class_criterion = criterions["class"]
        self.relation_criterion = criterions["relation"]

        self.c_opt = optimizers["c_opt"]
        self.d_opt = optimizers["d_opt"]
        self.r_opt = optimizers["r_opt"]

        self.src_loader = dataloaders["source_loader"]
        self.tar_loader = dataloaders["target_loader"]
        self.test_src_loader = dataloaders["test_src_loader"]
        self.test_tar_loader = dataloaders["test_tar_loader"]

        self.total_epoch = total_epoch
        self.log_interval = log_interval
        self.class_num = class_num
        self.img_size = 224
        self.d_iter = 3
        self.r_iter = 3

    def train(self):
        print("[Training]")

        for epoch in range(self.total_epoch):

            for index, (src, tar) in enumerate(zip(self.src_loader, self.tar_loader)):

                src_data, src_label = src
                tar_data, tar_label = tar

                size = min(src_data.shape[0], tar_data.shape[0])
                src_data, src_label = src_data[0:size], src_label[0:size]
                tar_data, tar_label = tar_data[0:size], tar_label[0:size]

                # label to long
                # src_label = src_label.type(torch.LongTensor)
                # tar_label = tar_label.type(torch.LongTensor)

                """ For MNIST """
                if src_data.shape[1] != 3:
                    src_data = src_data.expand(
                        src_data.shape[0], 3, self.img_size, self.img_size)

                src_data, src_label = src_data.cuda(), src_label.cuda()
                tar_data, tar_label = tar_data.cuda(), tar_label.cuda()

                """ train classifer """
                set_requires_grad(self.src_extractor, requires_grad=True)
                set_requires_grad(self.tar_extractor, requires_grad=True)
                set_requires_grad(self.relater, requires_grad=False)
                set_requires_grad(self.discriminator, requires_grad=False)

                src_z = self.src_extractor(src_data).cuda()
                tar_z = self.tar_extractor(tar_data).cuda()

                pred_class = self.classifier(src_z)
                class_loss = self.class_criterion(pred_class, src_label)

                rs = self.relater(src_z.detach())
                rs = 1-rs
                rs = rs/rs.sum()

                wasserstein_distance = self.discriminator(
                    alpha*rs*src_z).mean() - self.discriminator(tar_z).mean()

                sw = sliced_wasserstein_distance(
                    src_z, tar_z, num_projections=50)

                loss = class_loss + wasserstein_distance + 10*sw

                c_opt.zero_grad()
                loss.backward()
                c_opt.step()

                """ classify accuracy """
                _, predicted = torch.max(pred_class, 1)
                accuracy = 100.0 * \
                    (predicted == src_label).sum() / src_data.size(0)

                """ Validate Accuracy """
                with torch.no_grad():
                    pred_class_tar = self.classifier(tar_z.detach())
                    _, predicted_tar = torch.max(pred_class_tar, 1)
                    accuracy_tar = 100.0 * \
                        (predicted_tar == tar_label).sum()/tar_data.size(0)

                """ train relater """
                set_requires_grad(self.src_extractor, requires_grad=False)
                set_requires_grad(self.tar_extractor, requires_grad=False)
                set_requires_grad(self.relater, requires_grad=True)
                set_requires_grad(self.discriminator, requires_grad=False)

                with torch.no_grad():
                    src_z = self.src_extractor(src_data).detach()
                    tar_z = self.tar_extractor(tar_data).detach()

                for _ in range(self.r_iter):
                    src_r = self.relater(src_z)
                    tar_r = self.relater(tar_z)

                    r_src_loss = self.relation_criterion(
                        src_r, torch.ones(src_r.size(0), 1).type(torch.FloatTensor).cuda())

                    r_tar_loss = self.relation_criterion(
                        tar_r, torch.zeros(tar_r.size(0), 1).type(torch.FloatTensor).cuda())

                    r_loss = r_src_loss + r_tar_loss
                    self.r_opt.zero_grad()
                    r_loss.backward()
                    self.r_opt.step()

                """ train discriminator """

                set_requires_grad(self.src_extractor, requires_grad=False)
                set_requires_grad(self.tar_extractor, requires_grad=False)
                set_requires_grad(self.relater, requires_grad=False)
                set_requires_grad(self.discriminator, requires_grad=True)

                with torch.no_grad():
                    src_z = self.src_extractor(src_data)
                    tar_z = self.tar_extractor(tar_data)

                for _ in range(self.d_iter):
                    gp = gradient_penalty(self.discriminator, src_z, tar_z)

                    rs = self.relater(src_z).detach()
                    rs = 1-rs
                    rs = rs/rs.sum()

                    wasserstein_distance = self.discriminator(
                        alpha*rs*src_z).mean() - self.discriminator(tar_z).mean()

                    domain_loss = -wasserstein_distance + 10*gp
                    d_opt.zero_grad()
                    domain_loss.backward()
                    d_opt.step()

                if index % self.log_interval == 0:
                    print("[Epoch {:3d}] Total_loss: {:.4f}   C_loss: {:.4f}  R_loss: {:.4f}  D_loss:{:.4f}".format(
                        epoch, loss+r_loss+domain_loss, class_loss, r_loss, domain_loss))
                    print("Source Accuracy: {:.2f}".format(accuracy))
                    print("Target Accuracy: {:.2f}\n".format(accuracy_tar))

    def test(self):
        print("[Testing]")

        self.src_extractor.cuda().eval()
        self.tar_extractor.cuda().eval()
        self.classifier.cuda().eval()

        src_correct = 0
        tar_correct = 0

        # testing source
        for index, src in enumerate(self.test_src_loader):
            src_data, src_label = src
            # src_label = src_label.type(torch.LongTensor)
            src_data, src_label = src_data.cuda(), src_label.cuda()

            ''' for MNIST '''
            if src_data.shape[1] != 3:
                src_data = src_data.expand(
                    src_data.shape[0], 3, self.img_size, self.img_size)

            src_z = self.src_extractor(src_data)
            src_pred = self.classifier(src_z)
            _, predicted = torch.max(src_pred, 1)
            src_correct += (predicted == src_label).sum().item()

        # testing target
        for index, tar in enumerate(self.test_tar_loader):

            tar_data, tar_label = tar
            # tar_label = tar_label.type(torch.LongTensor)
            tar_data, tar_label = tar_data.cuda(), tar_label.cuda()

            tar_z = self.tar_extractor(tar_data)
            tar_pred = self.classifier(tar_z)
            _, predicted = torch.max(tar_pred, 1)
            tar_correct += (predicted == tar_label).sum().item()

        print("source accuracy: {:.2f}%".format(
            100*src_correct/len(self.test_src_loader.dataset)))
        print("target accuracy: {:.2f}%".format(
            100*tar_correct/len(self.test_tar_loader.dataset)))

    def save_model(self, path="./saved_WADA_Office/"):
        try:
            os.stat(path)
        except:
            os.mkdir(path)

        torch.save(self.src_extractor.state_dict(),
                   os.path.join(path, "WADA_E_SRC.pkl"))

        torch.save(self.tar_extractor.state_dict(),
                   os.path.join(path, "WADA_E_TAR.pkl"))

        torch.save(self.relater.state_dict(), os.path.join(path, "WADA_R.pkl"))

        torch.save(self.classifier.state_dict(),
                   os.path.join(path, "WADA_C.pkl"))

        torch.save(self.discriminator.state_dict(),
                   os.path.join(path, "WADA_D.pkl"))

    def load_model(self, path="./saved_WADA_Office/"):

        self.src_extractor.load_state_dict(
            torch.load(os.path.join(path, "WADA_E_SRC.pkl")))

        self.tar_extractor.load_state_dict(
            torch.load(os.path.join(path, "WADA_E_TAR.pkl")))

        self.relater.load_state_dict(
            torch.load(os.path.join(path, "WADA_R.pkl")))

        self.classifier.load_state_dict(
            torch.load(os.path.join(path, "WADA_C.pkl")))
        self.discriminator.load_state_dict(
            torch.load(os.path.join(path, "WADA_D.pkl")))

    def visualize(self, dim, plot_num):
        print("t-SNE reduces to dimension {}".format(dim))

        self.src_extractor.cpu().eval()
        self.tar_extractor.cpu().eval()

        src_data = torch.FloatTensor()
        tar_data = torch.FloatTensor()

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

        src_z = self.src_extractor(src_data)
        tar_z = self.tar_extractor(tar_data)

        # concate data, label information
        data = np.concatenate((src_z.detach().numpy(), tar_z.detach().numpy()))
        label = np.concatenate((src_label.numpy(), tar_label.numpy()))

        # concate domain information
        src_tag = torch.ones(src_z.size(0))
        tar_tag = torch.zeros(tar_z.size(0))
        tag = np.concatenate((src_tag.numpy(), tar_tag.numpy()))

        ''' t-SNE process '''
        tsne = TSNE(n_components=dim)

        embedding = tsne.fit_transform(data)
        embedding_max, embedding_min = np.max(
            embedding, 0), np.min(embedding, 0)
        embedding = (embedding-embedding_min) / (embedding_max - embedding_min)

        if dim == 2:
            visualize_2d("./saved_WADA_Office/", embedding,
                         label, tag, self.class_num)

        elif dim == 3:
            visualize_3d("./saved_WADA_Office/", embedding,
                         label, tag, self.class_num)


''' Unit test '''
if __name__ == "__main__":
    print("WADA model dev ver")

    batch_size = 100
    total_epoch = 100
    class_num = 10
    log_interval = 10
    src_partial = False
    tar_partial = False
    alpha = 100

    webcam = datasets.ImageFolder(
        "../dataset/office_caltech_10/webcam/", transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]))

    dslr = datasets.ImageFolder(
        "../dataset/office_caltech_10/dslr/", transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]))

    amazon = datasets.ImageFolder(
        "../dataset/office_caltech_10/amazon/", transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]))

    caltech = datasets.ImageFolder(
        "../dataset/office_caltech_10/caltech/", transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]))

    source_loader = torch.utils.data.DataLoader(
        dslr, batch_size=batch_size, shuffle=True)
    target_loader = torch.utils.data.DataLoader(
        amazon, batch_size=batch_size, shuffle=True)
    test_src_loader = source_loader
    test_tar_loader = target_loader
    """
    source_loader = get_office_loader(
        "amazon", partial=src_partial, batch_size=batch_size)
    target_loader = get_office_loader(
        "webcam", partial=tar_partial, batch_size=batch_size)
    test_src_loader = get_office_loader(
        "amazon", partial=src_partial, batch_size=batch_size)
    test_tar_loader = get_office_loader(
        "webcam", partial=tar_partial, batch_size=batch_size)
    """
    src_extractor = Extractor_Office().cuda()
    tar_extractor = Extractor_Office().cuda()

    classifier = Classifier_Office(class_num=class_num).cuda()

    relater = Relater_Office().cuda()
    discriminator = Discriminator_Office().cuda()

    class_criterion = nn.CrossEntropyLoss()
    relation_criterion = nn.BCELoss()

    '''
    c_opt = torch.optim.Adam([{"params": classifier.parameters()},
                              {"params": src_extractor.parameters()},
                              {"params": tar_extractor.parameters()}], lr=1e-4)

    d_opt = torch.optim.RMSprop(discriminator.parameters(), lr=1e-4)
    r_opt = torch.optim.RMSprop(relater.parameters(), lr=1e-4)
    '''

    c_opt = torch.optim.Adam([{"params": classifier.parameters()},
                              {"params": src_extractor.parameters()},
                              {"params": tar_extractor.parameters()}], lr=1e-3)

    d_opt = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    r_opt = torch.optim.Adam(relater.parameters(), lr=1e-4)

    components = {"src_extractor": src_extractor, "tar_extractor": src_extractor, "classifier": classifier,
                  "discriminator": discriminator, "relater": relater}
    optimizers = {"c_opt": c_opt, "d_opt": d_opt, "r_opt": r_opt}
    dataloaders = {"source_loader": source_loader, "target_loader": target_loader,
                   "test_src_loader": test_src_loader, "test_tar_loader": test_tar_loader}

    criterions = {"class": class_criterion, "relation": relation_criterion}

    model = WADA_II(components, optimizers, dataloaders, criterions,
                    total_epoch, class_num, log_interval)
    #model.train()
    #model.save_model()
    #model.load_model()
    #model.test()
    model.visualize(dim=2, plot_num=500)
    # model.visualize(dim=3)
    # model.load_model()
