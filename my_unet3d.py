from collections import OrderedDict

import torch
import torch.nn as nn
from math import floor

class UNet_5layer(nn.Module):

    #def __init__(self, in_channels=1, out_channels=2, init_features=8): # fÃ¼r nur 2 classen variante
    def __init__(self, in_channels=1, out_channels=2, init_features=16):
        super(UNet_5layer, self).__init__()

        features = init_features
        self.encoder1 = UNet_5layer._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2, ceil_mode=True)
        self.encoder2 = UNet_5layer._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = UNet_5layer._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = UNet_5layer._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder5 = UNet_5layer._block(features * 8, features * 16, name="enc5")
        self.pool5 = nn.MaxPool3d(kernel_size=2, stride=2)
        #self.encoder6 = UNet._block(features * 16, features * 32, name="enc4")
        #self.pool6 = nn.MaxPool3d(kernel_size=2, stride=2)


        self.bottleneck = UNet_5layer._block(features * 16, features * 32, name="bottleneck")

        #self.upconv6 = nn.ConvTranspose3d(
        #    features * 64, features * 32, kernel_size=2, stride=2
        #)
        #self.decoder6 = UNet._block((features * 32) * 2, features * 32, name="dec3")
        self.upconv5 = nn.ConvTranspose3d(
            features * 32, features * 16, kernel_size=2, stride=2
        )
        self.decoder5 = UNet_5layer._block((features * 16) * 2, features * 16, name="dec5")
        self.upconv4 = nn.ConvTranspose3d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet_5layer._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose3d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet_5layer._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose3d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet_5layer._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose3d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet_5layer._block(features * 2, features, name="dec1")


        self.conv = nn.Conv3d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

        self.dropout = nn.Dropout()


    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        enc5 = self.encoder5(self.pool4(enc4))
        #enc6 = self.encoder6(self.pool5(enc5))


        bottleneck = self.bottleneck(self.pool5(enc5))
        bottleneck = self.dropout(bottleneck)

        #dec6 = self.upconv6(bottleneck)
        #dec6 = torch.cat((dec6, enc6), dim=1)
        #dec6 = self.decoder6(dec6)

        dec5 = self.upconv5(bottleneck)
        dec5 = torch.cat((dec5,enc5), dim=1)
        dec5 = self.decoder5(dec5)

        dec4 = self.upconv4(dec5)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return (self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm3d(num_features=features)),
                    (name + "relu1", nn.SELU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm3d(num_features=features)),
                    (name + "relu2", nn.SELU(inplace=True)),
                ]
            )
        )


class UNet_4layer(nn.Module):

    #def __init__(self, in_channels=1, out_channels=2, init_features=8): # fÃ¼r nur 2 classen variante
    def __init__(self, in_channels=1, out_channels=2, init_features=16):
        super(UNet_4layer, self).__init__()

        features = init_features
        self.encoder1 = UNet_4layer._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2, ceil_mode=True)
        self.encoder2 = UNet_4layer._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = UNet_4layer._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = UNet_4layer._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        #self.encoder5 = UNet._block(features * 8, features * 16, name="enc5")
        #self.pool5 = nn.MaxPool3d(kernel_size=2, stride=2)
        #self.encoder6 = UNet._block(features * 16, features * 32, name="enc4")
        #self.pool6 = nn.MaxPool3d(kernel_size=2, stride=2)


        self.bottleneck = UNet_4layer._block(features * 8, features * 16, name="bottleneck")

        #self.upconv6 = nn.ConvTranspose3d(
        #    features * 64, features * 32, kernel_size=2, stride=2
        #)
        #self.decoder6 = UNet._block((features * 32) * 2, features * 32, name="dec3")
        #self.upconv5 = nn.ConvTranspose3d(
        #    features * 32, features * 16, kernel_size=2, stride=2
        #)
        #self.decoder5 = UNet._block((features * 16) * 2, features * 16, name="dec5")
        self.upconv4 = nn.ConvTranspose3d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet_4layer._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose3d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet_4layer._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose3d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet_4layer._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose3d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet_4layer._block(features * 2, features, name="dec1")


        self.conv = nn.Conv3d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

        self.dropout = nn.Dropout()


    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        #enc5 = self.encoder5(self.pool4(enc4))
        #enc6 = self.encoder6(self.pool5(enc5))


        bottleneck = self.bottleneck(self.pool4(enc4))
        bottleneck = self.dropout(bottleneck)

        #dec6 = self.upconv6(bottleneck)
        #dec6 = torch.cat((dec6, enc6), dim=1)
        #dec6 = self.decoder6(dec6)

        #dec5 = self.upconv5(bottleneck)
        #dec5 = torch.cat((dec5,enc5), dim=1)
        #dec5 = self.decoder5(dec5)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return (self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm3d(num_features=features)),
                    (name + "relu1", nn.SELU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm3d(num_features=features)),
                    (name + "relu2", nn.SELU(inplace=True)),
                ]
            )
        )

class UNet_3layer(nn.Module):

    #def __init__(self, in_channels=1, out_channels=2, init_features=8): # fÃ¼r nur 2 classen variante
    def __init__(self, in_channels=1, out_channels=2, init_features=16):
        super(UNet_3layer, self).__init__()

        features = init_features
        self.encoder1 = UNet_3layer._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2, ceil_mode=True)
        self.encoder2 = UNet_3layer._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = UNet_3layer._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        #self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        #self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        #self.encoder5 = UNet._block(features * 8, features * 16, name="enc5")
        #self.pool5 = nn.MaxPool3d(kernel_size=2, stride=2)
        #self.encoder6 = UNet._block(features * 16, features * 32, name="enc4")
        #self.pool6 = nn.MaxPool3d(kernel_size=2, stride=2)


        self.bottleneck = UNet_3layer._block(features * 4, features * 8, name="bottleneck")

        #self.upconv6 = nn.ConvTranspose3d(
        #    features * 64, features * 32, kernel_size=2, stride=2
        #)
        #self.decoder6 = UNet._block((features * 32) * 2, features * 32, name="dec3")
        #self.upconv5 = nn.ConvTranspose3d(
        #    features * 32, features * 16, kernel_size=2, stride=2
        #)
        #self.decoder5 = UNet._block((features * 16) * 2, features * 16, name="dec5")
        #self.upconv4 = nn.ConvTranspose3d(
        #    features * 16, features * 8, kernel_size=2, stride=2
        #)
        #self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose3d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet_3layer._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose3d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet_3layer._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose3d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet_3layer._block(features * 2, features, name="dec1")


        self.conv = nn.Conv3d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

        self.dropout = nn.Dropout()


    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        #enc4 = self.encoder4(self.pool3(enc3))
        #enc5 = self.encoder5(self.pool4(enc4))
        #enc6 = self.encoder6(self.pool5(enc5))


        bottleneck = self.bottleneck(self.pool3(enc3))
        bottleneck = self.dropout(bottleneck)

        #dec6 = self.upconv6(bottleneck)
        #dec6 = torch.cat((dec6, enc6), dim=1)
        #dec6 = self.decoder6(dec6)

    #    dec5 = self.upconv5(bottleneck)
        #dec5 = torch.cat((dec5,enc5), dim=1)
        #dec5 = self.decoder5(dec5)

    #    dec4 = self.upconv4(dec5)
        #dec4 = torch.cat((dec4, enc4), dim=1)
    #    dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return (self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm3d(num_features=features)),
                    (name + "relu1", nn.SELU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm3d(num_features=features)),
                    (name + "relu2", nn.SELU(inplace=True)),
                ]
            )
        )
