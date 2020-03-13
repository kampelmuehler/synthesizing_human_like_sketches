import torch.nn as nn
import torch.nn.functional as F
import torch
from utils.utils import adaptive_instance_normalization_mean_std as adaIN
import torch.utils.model_zoo as model_zoo
from utils.utils import chunks


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_conv = []
        self.encoder_conv.append(nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.encoder_conv.append(nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.encoder_conv.append(nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.encoder_conv.append(nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.encoder_conv.append(nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.encoder_conv.append(nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.encoder_conv.append(nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.encoder_conv.append(nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.encoder_conv.append(nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.encoder_conv.append(nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.encoder_conv.append(nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.encoder_conv.append(nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.encoder_conv.append(nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.encoder_batchnorm = []
        self.encoder_batchnorm.append(
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.encoder_batchnorm.append(
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.encoder_batchnorm.append(
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.encoder_batchnorm.append(
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.encoder_batchnorm.append(
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.encoder_batchnorm.append(
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.encoder_batchnorm.append(
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.encoder_batchnorm.append(
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.encoder_batchnorm.append(
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.encoder_batchnorm.append(
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.encoder_batchnorm.append(
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.encoder_batchnorm.append(
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.encoder_batchnorm.append(
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))

        # initialize encoder
        encoder_state_dict = model_zoo.load_url('https://download.pytorch.org/models/vgg16_bn-6c64b313.pth')
        encoder_weights = chunks([item[1] for item in list(encoder_state_dict.items()) if 'features' in item[0]], 6)
        for n, item in enumerate(encoder_weights):
            self.encoder_conv[n].weight.data = item[0]
            self.encoder_conv[n].bias.data = item[1]
            self.encoder_batchnorm[n].weight.data = item[2]
            self.encoder_batchnorm[n].bias.data = item[3]
            self.encoder_batchnorm[n].running_mean.data = item[4]
            self.encoder_batchnorm[n].running_var.data = item[5]

        self.encoder_conv = nn.ModuleList(self.encoder_conv)
        self.encoder_batchnorm = nn.ModuleList(self.encoder_batchnorm)

    def forward(self, x):
        x = self.encoder_conv[0](x)
        x = self.encoder_batchnorm[0](x)
        x = self.activation(x)
        encoder_conv1 = self.encoder_conv[1](x)
        x = self.encoder_batchnorm[1](encoder_conv1)
        x = self.activation(x)
        x = self.pool(x)
        x = self.encoder_conv[2](x)
        x = self.encoder_batchnorm[2](x)
        x = self.activation(x)
        encoder_conv3 = self.encoder_conv[3](x)
        x = self.encoder_batchnorm[3](encoder_conv3)
        x = self.activation(x)
        x = self.pool(x)
        x = self.encoder_conv[4](x)
        x = self.encoder_batchnorm[4](x)
        x = self.activation(x)
        x = self.encoder_conv[5](x)
        x = self.encoder_batchnorm[5](x)
        x = self.activation(x)
        encoder_conv6 = self.encoder_conv[6](x)
        x = self.encoder_batchnorm[6](encoder_conv6)
        x = self.activation(x)
        x = self.pool(x)
        x = self.encoder_conv[7](x)
        x = self.encoder_batchnorm[7](x)
        x = self.activation(x)
        x = self.encoder_conv[8](x)
        x = self.encoder_batchnorm[8](x)
        x = self.activation(x)
        encoder_conv9 = self.encoder_conv[9](x)
        x = self.encoder_batchnorm[9](encoder_conv9)
        x = self.activation(x)
        x = self.pool(x)
        x = self.encoder_conv[10](x)
        x = self.encoder_batchnorm[10](x)
        x = self.activation(x)
        x = self.encoder_conv[11](x)
        x = self.encoder_batchnorm[11](x)
        x = self.activation(x)
        x = self.encoder_conv[12](x)
        x = self.encoder_batchnorm[12](x)
        x = self.activation(x)
        return self.pool(x), encoder_conv1, encoder_conv3, encoder_conv6, encoder_conv9


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super(Decoder, self).__init__()
        self.activation = nn.ReLU(inplace=True)

        self.tconv1 = nn.ConvTranspose2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.tconv2 = nn.ConvTranspose2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.tconv3 = nn.ConvTranspose2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.tconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.tconv5 = nn.ConvTranspose2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.tconv6 = nn.ConvTranspose2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.tconv7 = nn.ConvTranspose2d(768, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.tconv8 = nn.ConvTranspose2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.tconv9 = nn.ConvTranspose2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.tconv10 = nn.ConvTranspose2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.tconv11 = nn.ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.tconv12 = nn.ConvTranspose2d(192, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.tconv13 = nn.ConvTranspose2d(64, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        # embedding layer for AdaIN means and stds @ bottleneck
        self.mean_embedding = nn.Embedding(num_classes, 512)
        self.std_embedding = nn.Embedding(num_classes, 512)
        # adaIN embeddings
        self.adaIN1_mean = nn.Embedding(num_classes, 512)
        self.adaIN2_mean = nn.Embedding(num_classes, 512)
        self.adaIN3_mean = nn.Embedding(num_classes, 512)
        self.adaIN4_mean = nn.Embedding(num_classes, 512)
        self.adaIN5_mean = nn.Embedding(num_classes, 512)
        self.adaIN6_mean = nn.Embedding(num_classes, 512)
        self.adaIN7_mean = nn.Embedding(num_classes, 256)
        self.adaIN8_mean = nn.Embedding(num_classes, 256)
        self.adaIN9_mean = nn.Embedding(num_classes, 256)
        self.adaIN10_mean = nn.Embedding(num_classes, 128)
        self.adaIN11_mean = nn.Embedding(num_classes, 128)
        self.adaIN12_mean = nn.Embedding(num_classes, 64)
        self.adaIN1_std = nn.Embedding(num_classes, 512)
        self.adaIN2_std = nn.Embedding(num_classes, 512)
        self.adaIN3_std = nn.Embedding(num_classes, 512)
        self.adaIN4_std = nn.Embedding(num_classes, 512)
        self.adaIN5_std = nn.Embedding(num_classes, 512)
        self.adaIN6_std = nn.Embedding(num_classes, 512)
        self.adaIN7_std = nn.Embedding(num_classes, 256)
        self.adaIN8_std = nn.Embedding(num_classes, 256)
        self.adaIN9_std = nn.Embedding(num_classes, 256)
        self.adaIN10_std = nn.Embedding(num_classes, 128)
        self.adaIN11_std = nn.Embedding(num_classes, 128)
        self.adaIN12_std = nn.Embedding(num_classes, 64)

    def forward(self, bottleneck, encoder_relu1, encoder_relu3, encoder_relu6, encoder_relu9, labels):
        # adaIN on bottleneck representation
        bottleneck = adaIN(bottleneck, self.mean_embedding(torch.argmax(labels, dim=1)),
                           self.std_embedding(torch.argmax(labels, dim=1)))

        #####################################################################
        # Decoder VGG16
        #####################################################################
        x = F.interpolate(bottleneck, scale_factor=2)

        #####################################################################
        # tconv1
        #####################################################################
        x = self.tconv1(x)
        x = adaIN(x, self.adaIN1_mean(torch.argmax(labels, dim=1)),
                  self.adaIN1_std(torch.argmax(labels, dim=1)))
        x = self.activation(x)

        #####################################################################
        # tconv2
        #####################################################################
        x = self.tconv2(x)
        x = adaIN(x, self.adaIN2_mean(torch.argmax(labels, dim=1)),
                  self.adaIN2_std(torch.argmax(labels, dim=1)))
        x = self.activation(x)

        #####################################################################
        # tconv3
        #####################################################################
        x = self.tconv3(x)
        x = adaIN(x, self.adaIN3_mean(torch.argmax(labels, dim=1)),
                  self.adaIN3_std(torch.argmax(labels, dim=1)))
        x = self.activation(x)
        x = F.interpolate(x, scale_factor=2)

        x = torch.cat((x, encoder_relu9), 1)

        #####################################################################
        # tconv4
        #####################################################################
        x = self.tconv4(x)
        x = adaIN(x, self.adaIN4_mean(torch.argmax(labels, dim=1)),
                  self.adaIN4_std(torch.argmax(labels, dim=1)))
        x = self.activation(x)

        #####################################################################
        # tconv5
        #####################################################################
        x = self.tconv5(x)
        x = adaIN(x, self.adaIN5_mean(torch.argmax(labels, dim=1)),
                  self.adaIN5_std(torch.argmax(labels, dim=1)))
        x = self.activation(x)

        #####################################################################
        # tconv6
        #####################################################################
        x = self.tconv6(x)
        x = adaIN(x, self.adaIN6_mean(torch.argmax(labels, dim=1)),
                  self.adaIN6_std(torch.argmax(labels, dim=1)))
        x = self.activation(x)
        x = F.interpolate(x, scale_factor=2)

        x = torch.cat((x, encoder_relu6), 1)
        #####################################################################
        # tconv7
        #####################################################################
        x = self.tconv7(x)
        x = adaIN(x, self.adaIN7_mean(torch.argmax(labels, dim=1)),
                  self.adaIN7_std(torch.argmax(labels, dim=1)))
        x = self.activation(x)

        #####################################################################
        # tconv8
        #####################################################################
        x = self.tconv8(x)
        x = adaIN(x, self.adaIN8_mean(torch.argmax(labels, dim=1)),
                  self.adaIN8_std(torch.argmax(labels, dim=1)))
        x = self.activation(x)

        #####################################################################
        # tconv9
        #####################################################################
        x = self.tconv9(x)
        x = adaIN(x, self.adaIN9_mean(torch.argmax(labels, dim=1)),
                  self.adaIN9_std(torch.argmax(labels, dim=1)))
        x = self.activation(x)
        x = F.interpolate(x, scale_factor=2)

        x = torch.cat((x, encoder_relu3), 1)

        #####################################################################
        # tconv10
        #####################################################################
        x = self.tconv10(x)
        x = adaIN(x, self.adaIN10_mean(torch.argmax(labels, dim=1)),
                  self.adaIN10_std(torch.argmax(labels, dim=1)))
        x = self.activation(x)

        #####################################################################
        # tconv11
        #####################################################################
        x = self.tconv11(x)
        x = adaIN(x, self.adaIN11_mean(torch.argmax(labels, dim=1)),
                  self.adaIN11_std(torch.argmax(labels, dim=1)))
        x = self.activation(x)
        x = F.interpolate(x, scale_factor=2)

        x = torch.cat((x, encoder_relu1), 1)

        #####################################################################
        # tconv12
        #####################################################################
        x = self.tconv12(x)
        x = adaIN(x, self.adaIN12_mean(torch.argmax(labels, dim=1)),
                  self.adaIN12_std(torch.argmax(labels, dim=1)))
        x = self.activation(x)

        #####################################################################
        # tconv13
        #####################################################################
        reconstructed = self.tconv13(x)
        reconstructed = torch.sigmoid(reconstructed)
        return reconstructed


class SketchGenerator(nn.Module):
    def __init__(self, num_classes=125):
        super(SketchGenerator, self).__init__()
        self.encoder = Encoder()
        # freeze encoder weights
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.decoder = Decoder(num_classes=num_classes)

    def forward(self, x, labels):
        x = self.encoder(x)
        return self.decoder(*x, labels)


if __name__ == '__main__':
    SketchGenerator()
