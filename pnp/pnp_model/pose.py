
from torchvision.models import resnet18, ResNet18_Weights
import torch
import torch.nn as nn


# use: use model.train() or model.eval() to switch between training and evaluation mode as appropriate
class Conv(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # pre-trained model
        self.weights = ResNet50_Weights.IMAGENET1K_V1
        self.preprocess = self.weights.transforms(antialias=True)   # usage: img_transformed = preprocess(img)
        model = resnet50(weights=self.weights)                      # input:  (bs, 3, 224, 224)
                                                                    # output: (1000)

        # model._modules.keys()
        # odict_keys(['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc'])
        self.conv1 = model._modules['conv1']                        # output: (bs, 64, 112, 112)
        self.bn1   = model._modules['bn1']                          # output: (bs, 64, 112, 112)
        
        self.relu    = model._modules['relu']                       # output: (bs, 64, 112, 112)
        self.maxpool = model._modules['maxpool']                    # output: (bs, 64,  56,  56)
        self.layer1  = model._modules['layer1']                     # output: (bs, 256, 56, 56)
        self.layer2  = model._modules['layer2']                     # output: (bs, 512, 28, 28)
        self.layer3  = model._modules['layer3']                     # output: (bs, 1024, 14, 14)
        self.layer4  = model._modules['layer4']                     # output: (bs, 2048, 7, 7)
        self.avgpool = model._modules['avgpool']                    # output: (bs, 2048, 1, 1)
        self.fc      = model._modules['fc']                         # output: (bs, 1000)
        del model
        
        self.relu    = nn.ReLU(inplace=True)                        # output: (bs, 1000)
        self.dense   = nn.Linear(1000, 4)                           # output: (bs, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x) 
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.relu(x)
        x = self.dense(x)
        return x



from torchvision.models import resnet18, ResNet18_Weights
import torch
import torch.nn as nn


# use: use model.train() or model.eval() to switch between training and evaluation mode as appropriate
class Resnet18(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # pre-trained model
        self.weights = ResNet18_Weights.IMAGENET1K_V1
        self.preprocess = self.weights.transforms(antialias=True)   # usage: img_transformed = preprocess(img)
        model = resnet18(weights=self.weights)                      # input:  (bs, 3, 224, 224)
                                                                    # output: (1000)

        # model._modules.keys()
        # odict_keys(['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc'])
        self.conv1 = model._modules['conv1']                 # output: (bs, 64, 112, 112)
        self.bn1   = model._modules['bn1']                   # output: (bs, 64, 112, 112)
        
        self.relu    = model._modules['relu']                # output: (bs, 64, 112, 112)
        self.maxpool = model._modules['maxpool']             # output: (bs, 64,  56,  56)
        self.layer1  = model._modules['layer1']              # output: (bs, 256, 56, 56)
        self.layer2  = model._modules['layer2']              # output: (bs, 512, 28, 28)
        self.layer3  = model._modules['layer3']              # output: (bs, 1024, 14, 14)
        self.layer4  = model._modules['layer4']              # output: (bs, 2048, 7, 7)
        del model

        self.conv2   = nn.Conv2d(2048, 1024, kernel_size=1)  # output: (bs, 1024, 7, 7)
        self.bn2     = nn.BatchNorm2d(1024)                  # output: (bs, 1024, 7, 7)
        self.relu    = nn.ReLU()                             # output: (bs, 1024, 7, 7)
        self.conv3   = nn.Conv2d(1024, 256, kernel_size=1)   # output: (bs, 256, 7, 7)
        self.bn3     = nn.BatchNorm2d(256)                   # output: (bs, 256, 7, 7)
        self.relu    = nn.ReLU()                             # output: (bs, 256, 7, 7)
        self.conv4   = nn.Conv2d(256, 64, kernel_size=1)     # output: (bs, 64, 7, 7)
        self.bn4     = nn.BatchNorm2d(64)                    # output: (bs, 64, 7, 7)
        self.relu    = nn.ReLU()                             # output: (bs, 64, 7, 7)
        self.conv5   = nn.Conv2d(64, 16, kernel_size=1)      # output: (bs, 16, 7, 7)
        self.bn5     = nn.BatchNorm2d(16)                    # output: (bs, 16, 7, 7)
        self.relu    = nn.ReLU()                             # output: (bs, 16, 7, 7)
        
        self.dense   = nn.Linear(784, 4)                     # output: (bs, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x) 
        x = self.layer4(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x) 

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x) 

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x) 

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x) 

        x = x.view(-1, 784)
        x = self.dense(x)
        return x






from torchvision.models import resnet50, ResNet50_Weights
import torch
import torch.nn as nn


# use: use model.train() or model.eval() to switch between training and evaluation mode as appropriate
class Resnet50(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # pre-trained model
        self.weights = ResNet50_Weights.IMAGENET1K_V1
        self.preprocess = self.weights.transforms(antialias=True)   # usage: img_transformed = preprocess(img)
        model = resnet50(weights=self.weights)                      # input:  (bs, 3, 224, 224)
                                                                    # output: (1000)

        # model._modules.keys()
        # odict_keys(['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc'])
        self.conv1 = model._modules['conv1']                 # output: (bs, 64, 112, 112)
        self.bn1   = model._modules['bn1']                   # output: (bs, 64, 112, 112)
        
        self.relu    = model._modules['relu']                # output: (bs, 64, 112, 112)
        self.maxpool = model._modules['maxpool']             # output: (bs, 64,  56,  56)
        self.layer1  = model._modules['layer1']              # output: (bs, 256, 56, 56)
        self.layer2  = model._modules['layer2']              # output: (bs, 512, 28, 28)
        self.layer3  = model._modules['layer3']              # output: (bs, 1024, 14, 14)
        self.layer4  = model._modules['layer4']              # output: (bs, 2048, 7, 7)
        del model

        self.conv2   = nn.Conv2d(2048, 1024, kernel_size=1)  # output: (bs, 1024, 7, 7)
        self.bn2     = nn.BatchNorm2d(1024)                  # output: (bs, 1024, 7, 7)
        self.relu    = nn.ReLU()                             # output: (bs, 1024, 7, 7)
        self.conv3   = nn.Conv2d(1024, 256, kernel_size=1)   # output: (bs, 256, 7, 7)
        self.bn3     = nn.BatchNorm2d(256)                   # output: (bs, 256, 7, 7)
        self.relu    = nn.ReLU()                             # output: (bs, 256, 7, 7)
        self.conv4   = nn.Conv2d(256, 64, kernel_size=1)     # output: (bs, 64, 7, 7)
        self.bn4     = nn.BatchNorm2d(64)                    # output: (bs, 64, 7, 7)
        self.relu    = nn.ReLU()                             # output: (bs, 64, 7, 7)
        self.conv5   = nn.Conv2d(64, 16, kernel_size=1)      # output: (bs, 16, 7, 7)
        self.bn5     = nn.BatchNorm2d(16)                    # output: (bs, 16, 7, 7)
        self.relu    = nn.ReLU()                             # output: (bs, 16, 7, 7)
        
        self.dense   = nn.Linear(784, 4)                     # output: (bs, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x) 
        x = self.layer4(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x) 

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x) 

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x) 

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x) 

        x = x.view(-1, 784)
        x = self.dense(x)
        return x
