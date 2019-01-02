import torch
import torch.nn as nn
from torch.autograd import grad


def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


def gradient_penalty(critic, h_s, h_t):
    ''' Gradeitnt penalty for Wasserstein GAN'''
    alpha = torch.rand(h_s.size(0), 1).cuda()
    differences = h_t - h_s
    interpolates = h_s + (alpha * differences)
    interpolates = torch.cat([interpolates, h_s, h_t]).requires_grad_()
    # interpolates.requires_grad_()
    preds = critic(interpolates)
    gradients = grad(preds, interpolates,
                     grad_outputs=torch.ones_like(preds),
                     retain_graph=True, create_graph=True)[0]
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1)**2).mean()

    return gradient_penalty


class GradReverse(torch.autograd.Function):
    '''
    Gradient Reversal Layer
    '''
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg()*ctx.constant
        return grad_output, None

    # pylint raise E0213 warning here
    def grad_reverse(x, constant):
        '''
        Extension of grad reverse layer
        '''
        return GradReverse.apply(x, constant)


class Extractor_new(nn.Module):

    def __init__(self, in_channels=64, lrelu_slope=0.2, encoded_dim=100):
        super(Extractor_new, self).__init__()

        self.in_channels = in_channels
        self.lrelu_slope = lrelu_slope
        self.encoded_dim = encoded_dim

        self.extract = nn.Sequential(
            nn.Conv2d(3, self.in_channels, 5),
            nn.BatchNorm2d(self.in_channels),
            nn.MaxPool2d(2),
            nn.LeakyReLU(self.lrelu_slope),
            nn.Conv2d(self.in_channels, self.in_channels//2, 5),
            nn.BatchNorm2d(self.in_channels//2),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.LeakyReLU(self.lrelu_slope)
        )

    def forward(self, x):
        z = self.extract(x)
        z = z.view(-1, 32*4*4)
        return z


class Extractor(nn.Module):
    ''' Feature extractor '''

    def __init__(self, in_channels=16, lrelu_slope=0.2, encoded_dim=100):
        super(Extractor, self).__init__()

        self.in_channels = in_channels
        self.lrelu_slope = lrelu_slope

        self.encoded_dim = encoded_dim

        self.extract = nn.Sequential(
            nn.Conv2d(3, self.in_channels*1, kernel_size=5, padding=1),
            nn.BatchNorm2d(self.in_channels*1),
            nn.MaxPool2d(2),
            nn.LeakyReLU(self.lrelu_slope),
            # nn.ReLU(),
            nn.Conv2d(self.in_channels*1, self.in_channels *
                      4, kernel_size=5, padding=1),
            nn.BatchNorm2d(self.in_channels*4),
            # added
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            # nn.LeakyReLU(self.lrelu_slope)
        )

        self.fc = nn.Sequential(
            nn.Linear(self.in_channels*4*5*5, self.encoded_dim),
            nn.ReLU()
        )

    def forward(self, x):
        z = self.extract(x)
        z = z.view(-1, 64*5*5)
        #z = self.fc(z)

        return z


class Classifier(nn.Module):
    ''' Task Classifier '''

    def __init__(self, encoded_dim=100, class_num=10):
        super(Classifier, self).__init__()

        self.encoded_dim = encoded_dim
        self.class_num = class_num

        self.classify = nn.Sequential(
            #nn.Linear(self.encoded_dim, 100),
            #nn.Linear(64*5*5, 100),
            nn.Linear(32*4*4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            # added
            nn.Dropout(),
            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Linear(50, self.class_num),
            nn.LogSoftmax(1)
        )

    def forward(self, x):
        return self.classify(x)


class Discriminator(nn.Module):
    ''' Domain Discriminator '''

    def __init__(self, encoded_dim):
        super(Discriminator, self).__init__()
        self.encoded_dim = encoded_dim

        self.classify = nn.Sequential(
            #nn.Linear(self.encoded_dim, 64),
            nn.Linear(32*4*4, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.LogSoftmax(1)
        )

    def forward(self, x):
        return self.classify(x)


class Discriminator_WGAN(nn.Module):
    ''' Domain Discriminator '''

    def __init__(self, encoded_dim):
        super(Discriminator_WGAN, self).__init__()
        self.encoded_dim = encoded_dim

        self.classify = nn.Sequential(
            #nn.Linear(self.encoded_dim, 64),
            #nn.Linear(64*5*5, 64),
            nn.Linear(32*4*4, 64),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),
            # nn.Softmax(1)
        )

    def forward(self, x):
        return self.classify(x)


class Discriminator_GRL(nn.Module):
    ''' Domain Discriminator with gradient reversal layer'''

    def __init__(self, encoded_dim):
        super(Discriminator_GRL, self).__init__()
        self.encoded_dim = encoded_dim

        self.classify = nn.Sequential(
            #nn.Linear(self.encoded_dim, 64),
            #nn.Linear(64*5*5, 64),
            nn.Linear(32*4*4, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.LogSoftmax(1)
        )

    def forward(self, x, constant):
        x = GradReverse.grad_reverse(x, constant)
        return self.classify(x)


class Discriminator_mini(nn.Module):
    ''' class level discriminator for SAN'''

    def __init__(self, encoded_dim):
        self.encoded_dim = encoded_dim

        self.classify = nn.Sequential(

            nn.Linear(encoded_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.classify(x)


class Relater(nn.Module):

    ''' Relater network used in WADA model '''

    def __init__(self, encoded_dim):
        super(Relater, self).__init__()
        self.encoded_dim = encoded_dim

        self.distinguish = nn.Sequential(
            #nn.Linear(64*5*5, 100),
            nn.Linear(32*4*4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.distinguish(x)


class AutoEncoder(nn.Module):

    def __init__(self, in_channels=16, lrelu_slope=0.2):
        super(AutoEncoder, self).__init__()
        self.in_channels = in_channels
        self.lrelu_slope = lrelu_slope

        self.encoder = nn.Sequential(
            nn.Conv2d(3, self.in_channels, 5),
            nn.BatchNorm2d(self.in_channels),
            nn.MaxPool2d(2),
            nn.LeakyReLU(self.lrelu_slope),
            nn.Conv2d(self.in_channels, self.in_channels//2, 5),
            nn.BatchNorm2d(self.in_channels//2),
            nn.Dropout2d(),
            nn.MaxPool2d(2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.in_channels//2, self.in_channels, 5),
            nn.LeakyReLU(self.lrelu_slope),
            nn.ConvTranspose2d(self.in_channels, ),
        )

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
