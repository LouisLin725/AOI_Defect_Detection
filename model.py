import torchvision
import torch.nn as nn


class LeNet5(nn.Module):
    # Initializing the model structure
    def __init__(self):
        super(LeNet5, self).__init__()

        in_channels = 3
        num_classes = 6
        self.feature_channels = 120 * 50 * 50

        # Convolution 1st_block, input_shape=(3,224,224)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=6,
                kernel_size=5,
                stride=1,
                padding=2,
                bias=True
            ),  # shape -> (6, 224, 224)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # shape -> (6, 112, 112)
        )

        # Convolution 2nd_block, input_shape=(6, 112, 112)
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5, 1, 0, bias=True),  # shape -> (16, 108, 108)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # shape -> (16, 54, 54)
        )

        # Convolution 3rd_block, input_shape=(16, 54, 54)
        self.conv3 = nn.Conv2d(16, 120, 5, 1, 0, bias=True)  # shape -> (120, 50, 50)

        # Fully connected 1, #input_shape=(120*50*50), #output_shape=84
        self.fc1 = nn.Sequential(
            nn.Linear(self.feature_channels, 84),
            nn.ReLU(inplace=True)
        )

        # Output layer, Fully connected 2, #input_shape=(84), #output_shape=6
        self.fc2 = nn.Linear(84, num_classes)

    # forward pass
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, self.feature_channels)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


# Class of the pre-trained VGG16 model
class VGG16_pre(nn.Module):
    # Initializing the model structure
    def __init__(self, pretrained=True, batch_norm=True):
        super(VGG16_pre, self).__init__()

        # Set number of model output and feature channel according to your dataset
        self.in_channels = 3
        num_classes = 6
        feature_channels = 512 * 7 * 7

        # Using feature extractor of pretrained model
        print('Use pretrained VGG feature extractor')
        if batch_norm:
            self.feature = torchvision.models.vgg16_bn(pretrained=True).features
            self.feature = nn.Sequential(*list(self.feature.children())[:-1])  # Remove pool5
        else:
            self.feature = torchvision.models.vgg16(pretrained=True).features
            self.feature = nn.Sequential(*list(self.feature.children())[:-1])  # Remove pool5

        self.fc = nn.Linear(feature_channels, num_classes)

    # forward pass
    def forward(self, x):
        x = self.feature(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.fc(x)

        return y


# Class of the revised VGG16 model
class VGG16_revised(nn.Module):
    # Initializing the model structure
    def __init__(self):
        super(VGG16_revised, self).__init__()

        # Convolution 1st_block, input_shape=(3,224,224)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),  # shape -> (64, 224, 224)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # shape -> (64, 112, 112)
        )

        # Convolution 2nd_block, input_shape=(64, 112, 112)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),  # shape -> (32, 112, 112)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # shape -> (32, 56, 56)
        )

        # Convolution 3rd_block, input_shape=(32, 56, 56)
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, 1),  # shape -> (16, 56, 56)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # shape -> (16, 28, 28)
        )

        # Convolution 4th_block, input_shape=(16, 28, 28)
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 8, 3, 1, 1),  # shape -> (8, 28, 28)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # shape -> (8, 14, 14)
        )

        # Convolution 5th_block, input_shape=(8, 14, 14)
        self.conv5 = nn.Sequential(
            nn.Conv2d(8, 8, 3, 1, 1),  # shape -> (8, 14, 14)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # shape -> (8, 7, 7)

        )

        # Fully connected 1 ,#input_shape=(8*7*7), #output_shape=1024
        self.fc1 = nn.Linear(8 * 7 * 7, 1024)

        # Fully connected 2 ,#input_shape=(1024), #output_shape=6
        self.fc2 = nn.Linear(1024, 6)

    # forward pass
    def forward(self, x):
        out = self.conv1(x)  # shape -> (batch, 64, 112, 112)
        out = self.conv2(out)  # shape -> (batch, 32, 56, 56)
        out = self.conv3(out)  # shape -> (batch, 16, 28, 28)
        out = self.conv4(out)  # shape -> (batch, 8, 14, 14)
        out = self.conv5(out)  # shape -> (batch, 8, 7, 7)
        out = out.view(out.size(0), -1)  # shape -> (batch, 8*7*7)
        out = self.fc1(out)  # shape -> (batch, 1024)
        out = self.fc2(out)  # shape -> (batch, 6)

        return out