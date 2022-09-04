import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Union



class AlexNetBN(nn.Module):
    def __init__(self, input_channels=3, num_classes=1000, cfg=[128,128,128,128,128]):
        super(AlexNetBN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, cfg[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(cfg[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(cfg[0], cfg[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(cfg[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(cfg[1], cfg[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(cfg[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(cfg[2], cfg[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(cfg[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(cfg[3], cfg[4], kernel_size=3, padding=1),
            nn.BatchNorm2d(cfg[4]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Linear(cfg[4] * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    
    def getLatents(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier[0](x)
        return x
    
    
    
class Net(nn.Module):
    def __init__(self, input_channels=3, num_classes=1000):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    
    def getLatents(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier[0](x)
        return x
    
    
    
class AENet(nn.Module):
    def __init__(self, input_channels=3):
        super(AENet, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2)
        
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding = 1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding = 1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding = 1)
        
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size = 2, stride = 2)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size = 2, stride = 2)
        self.deconv3 = nn.ConvTranspose2d(32, input_channels, kernel_size = 2, stride = 2)
        
        self.sig = nn.Sigmoid()

        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    
    def encoder(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        return x
    
    
    def decoder(self, x):
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.sig(self.deconv3(x))
        return x
    
    
    
class Generator(nn.Module):
    def __init__(self, input_dim=100, out_channels=3):
        super(Generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(input_dim, 1024, 4, stride=1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.deconv2 = nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.deconv3 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.deconv4 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.deconv5 = nn.ConvTranspose2d(128, out_channels, 4, stride=2, padding=1)
        
        self.relu = nn.ReLU()
        self.tan = nn.Tanh()
        
        
    def forward(self, x):
        x = self.relu(self.bn1(self.deconv1(x)))
        x = self.relu(self.bn2(self.deconv2(x)))
        x = self.relu(self.bn3(self.deconv3(x)))
        x = self.relu(self.bn4(self.deconv4(x)))
        x = self.tan(self.deconv5(x))
        return x

    
    
class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 512, 4, stride=1)
        
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.lrelu = nn.LeakyReLU(0.2)
        
        
    def forward(self, x):
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.bn2(self.conv2(x)))
        x = self.lrelu(self.bn3(self.conv3(x)))
        x = self.lrelu(self.bn4(self.conv4(x)))
        x = self.lrelu(self.conv5(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    
    
class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

        
    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return (output1, output2)

    
    def get_embedding(self, x):
        return self.embedding_net(x)
    

    
class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

        
    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return (output1, output2, output3)

    
    def get_embedding(self, x):
        return self.embedding_net(x)
    
    
    
# downConv upConv finalCon for Unet
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        
    def forward(self, x):
        return self.double_conv(x)


    
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

        
    def forward(self, x):
        return self.maxpool_conv(x)

    

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

            
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

    

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        
    def forward(self, x):
        return self.conv(x)

    
    
class UNet(nn.Module):
    def __init__(self, input_channels, bilinear=False):
        super(UNet, self).__init__()
        self.inc = DoubleConv(input_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, input_channels)

        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
    
    def encoder(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        return x
    
    
    
class Ensemble(nn.Module):
    def __init__(self, models):
        super(Ensemble, self).__init__()
        for idx, model in enumerate(models):
            setattr(self, f"model{idx}", model)
        self.num_models = len(models)
    
    
    def forward(self, x):
        y = 0
        for name, model in self.named_children():
            y += model(x)
        return y
    
    
    def add_model(self, model):
        setattr(self, f"model{self.num_models}", model)
        self.num_models += 1
        
        
    def nograd_per_model(self, model):
        for param in model.parameters():
            param.requires_grad = False
    
    
    def nograd_models(self):
        for name, model in self.named_children():
            self.nograd_per_model(model)
            
            
    def get_models(self):
        models = []
        for name, model in self.named_children():
            models.append(model)
        return models
    
    

class EfficientNet(nn.Module):
    def __init__(
        self,
        efficientnet_fn,
        final_dim,
    ):
        super().__init__()
        self.efficientnet = efficientnet_fn(pretrained=True)
        lastconv_output_channels = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            self.efficientnet.classifier[0],
            nn.Linear(lastconv_output_channels, final_dim),
        )
        
    def forward(self, x):
        x = self.efficientnet(x)
        return x