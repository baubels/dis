
from torchvision.models import resnet50, ResNet50_Weights
import torch, torch.nn as nn


# use: use model.train() or model.eval() to switch between training and evaluation mode as appropriate
class Resnet50PAD(nn.Module): # ~23 mil params
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
        self.dense   = nn.Linear(1000, 100)                         # output: (bs, 22)
        self.relu2   = nn.ReLU(inplace=True)                        # output: (bs, 1000)
        self.dense2  = nn.Linear(100,   22)                         # output: (bs, 5)

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
        x = self.relu2(x)
        x = self.dense2(x)  # the fifth component yields the location of the 2d keypoints
        return x


# from torchvision.models import resnet18, ResNet18_Weights
# class Resnet18PAD(nn.Module): # ~11mil params
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)

#         # pre-trained model
#         self.weights = ResNet18_Weights.IMAGENET1K_V1
#         self.preprocess = self.weights.transforms(antialias=True)   # usage: img_transformed = preprocess(img)
#         model = resnet18(weights=self.weights)                      # input:  (bs, 3, 224, 224)
#                                                                     # output: (1000)

#         # model._modules.keys()
#         # odict_keys(['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc'])
#         self.conv1 = model._modules['conv1']                        # output: (bs, 64, 112, 112)
#         self.bn1   = model._modules['bn1']                          # output: (bs, 64, 112, 112)
        
#         self.relu    = model._modules['relu']                       # output: (bs, 64, 112, 112)
#         self.maxpool = model._modules['maxpool']                    # output: (bs, 64,  56,  56)
#         self.layer1  = model._modules['layer1']                     # output: (bs, 256, 56, 56)
#         self.layer2  = model._modules['layer2']                     # output: (bs, 512, 28, 28)
#         self.layer3  = model._modules['layer3']                     # output: (bs, 1024, 14, 14)
#         self.layer4  = model._modules['layer4']                     # output: (bs, 2048, 7, 7)
#         self.avgpool = model._modules['avgpool']                    # output: (bs, 2048, 1, 1)
#         self.fc      = model._modules['fc']                         # output: (bs, 1000)
#         del model
        
#         self.relu    = nn.ReLU(inplace=True)                        # output: (bs, 1000)
#         self.dense   = nn.Linear(1000, 5)                           # output: (bs, 4)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x) 
#         x = self.layer4(x)

#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#         x = self.relu(x)
#         x = self.dense(x) # the fifth component is distance
#         return x


# from torchvision.models import resnet18, ResNet18_Weights
# class Resnet18PADv2(nn.Module): # ~11mil params
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)

#         # pre-trained model
#         self.weights = ResNet18_Weights.IMAGENET1K_V1
#         self.preprocess = self.weights.transforms(antialias=True)   # usage: img_transformed = preprocess(img)
#         model = resnet18(weights=self.weights)                      # input:  (bs, 3, 224, 224)
#                                                                     # output: (1000)

#         # model._modules.keys()
#         # odict_keys(['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc'])
#         self.conv1 = model._modules['conv1']                        # output: (bs, 64, 112, 112)
#         self.bn1   = model._modules['bn1']                          # output: (bs, 64, 112, 112)
        
#         self.relu    = model._modules['relu']                       # output: (bs, 64, 112, 112)
#         self.maxpool = model._modules['maxpool']                    # output: (bs, 64,  56,  56)
#         self.layer1  = model._modules['layer1']                     # output: (bs, 256, 56, 56)
#         self.layer2  = model._modules['layer2']                     # output: (bs, 512, 28, 28)
#         self.layer3  = model._modules['layer3']                     # output: (bs, 1024, 14, 14)
#         self.layer4  = model._modules['layer4']                     # output: (bs, 2048, 7, 7)
#         self.avgpool = model._modules['avgpool']                    # output: (bs, 2048, 1, 1)
#         self.fc      = model._modules['fc']                         # output: (bs, 1000)
#         del model
        
#         self.relu    = nn.ReLU(inplace=True)                        # output: (bs, 1000)
#         self.dense   = nn.Linear(1000, 5)                           # output: (bs, 4)
#         self.tanh = nn.Tanh()               # <- only to be applied on the first 4 elements
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x) 
#         x = self.layer4(x)

#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#         x = self.relu(x)
#         x = self.dense(x) # the fifth component is distance
#         x[:,:4] = self.tanh(x[:,:4])            # is there any advantage to forcing quaternion outputs to be within [-1,1]?
#         x[:,4] = 20*self.sigmoid(x[:,4])        # and making the sigmoid to be within [0, 20]?

#         return x

# class AlexNetPAD(nn.Module): # 62 mil params
#     """
#     Neural network model consisting of layers proposed by the AlexNet paper.
#     Courtesy of: `https://github.com/dansuh17/alexnet-pytorch`
#     """
#     def __init__(self, num_classes=5):
#         """
#         Define and allocate layers for this neural net.
#         Args:
#             num_classes (int): number of classes to predict with this model
#         """
#         super().__init__()
#                                                                                              # input size should be : (b x 3 x 224 x 224)
#         # The image in the original paper states that width and height are 224 pixels, but
#         # the dimensions after first convolution layer do not lead to 55 x 55.

#         self.net = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2),  # (b x  96 x 55 x 55)
#             nn.ReLU(),
#             nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # section 3.3
#             nn.MaxPool2d(kernel_size=3, stride=2),                                           # (b x  96 x 27 x 27)
#             nn.Conv2d(96, 256, 5, padding=2),                                                # (b x 256 x 27 x 27)
#             nn.ReLU(),
#             nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
#             nn.MaxPool2d(kernel_size=3, stride=2),                                           # (b x 256 x 13 x 13)
#             nn.Conv2d(256, 384, 3, padding=1),                                               # (b x 384 x 13 x 13)
#             nn.ReLU(),
#             nn.Conv2d(384, 384, 3, padding=1),                                               # (b x 384 x 13 x 13)
#             nn.ReLU(),
#             nn.Conv2d(384, 256, 3, padding=1),                                               # (b x 256 x 13 x 13)
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=2),                                           # (b x 256 x 6 x 6)
#         )

#         # classifier is just a name for linear layers
#         # this is a lot of params; maybe it can be made smaller?
#         self.classifier = nn.Sequential( # ~58 mil params
#             nn.Dropout(p=0.5, inplace=True),
#             nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
#             nn.ReLU(),
#             nn.Dropout(p=0.5, inplace=True),
#             nn.Linear(in_features=4096, out_features=4096),
#             nn.ReLU(),
#             nn.Linear(in_features=4096, out_features=num_classes),
#         )
#         self.init_bias()  # initialize bias

#     def init_bias(self):
#         for layer in self.net:
#             if isinstance(layer, nn.Conv2d):
#                 nn.init.normal_(layer.weight, mean=0, std=0.01)
#                 nn.init.constant_(layer.bias, 0)

#         # original paper = 1 for Conv2d layers 2nd, 4th, and 5th conv layers
#         nn.init.constant_(self.net[4].bias, 1)
#         nn.init.constant_(self.net[10].bias, 1)
#         nn.init.constant_(self.net[12].bias, 1)

#     def forward(self, x):
#         """
#         Pass the input through the net.
#         Args:
#             x (Tensor): input tensor
#         Returns:
#             output (Tensor): output tensor
#         """
#         x = self.net(x)
#         x = x.view(-1, 256 * 6 * 6)  # reduce the dimensions for linear layer input
#         x = self.classifier(x)
#         return x


# from torchvision.models import resnet18, ResNet18_Weights
# class Custom1PAD(nn.Module): # 817077 params
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)

#         # pre-trained model
#         self.weights = ResNet18_Weights.IMAGENET1K_V1
#         self.preprocess = self.weights.transforms(antialias=True)   # usage: img_transformed = preprocess(img)
#         model = resnet18(weights=self.weights)                      # input:  (bs, 3, 224, 224)
#                                                                     # output: (1000)

#         # model._modules.keys()
#         # odict_keys(['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc'])
#         self.conv1 = model._modules['conv1']                        # output: (bs, 64, 112, 112)
#         self.bn1   = model._modules['bn1']                          # output: (bs, 64, 112, 112)
        
#         self.relu    = model._modules['relu']                       # output: (bs, 64, 112, 112)
#         self.maxpool = model._modules['maxpool']                    # output: (bs, 64,  56,  56)
#         self.layer1  = model._modules['layer1']                     # output: (bs, 256, 56, 56)
#         self.layer2  = model._modules['layer2']                     # output: (bs, 512, 28, 28)
#         self.avgpool = model._modules['avgpool']                    # output: (bs, 128, 1, 1)
#         self.fc      = nn.Linear(128, 1000)                         # output: (bs, 1000)
#         del model

#         self.relu    = nn.ReLU(inplace=True)                        # output: (bs, 1000)
#         self.dense   = nn.Linear(1000, 5)                           # output: (bs, 4)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)

#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#         x = self.relu(x)
#         x = self.dense(x) # the fifth component is distance
#         return x


# from torchvision.models import resnet18, ResNet18_Weights
# class Custom1PADv2(nn.Module): # 817077 params
#     """adding an activation function to the depth preds lets training last for longer than 1 epoch before NaN'ing."""
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)

#         # pre-trained model
#         self.weights = ResNet18_Weights.IMAGENET1K_V1
#         self.preprocess = self.weights.transforms(antialias=True)   # usage: img_transformed = preprocess(img)
#         model = resnet18(weights=self.weights)                      # input:  (bs, 3, 224, 224)
#                                                                     # output: (1000)

#         # model._modules.keys()
#         # odict_keys(['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc'])
#         self.conv1 = model._modules['conv1']                        # output: (bs, 64, 112, 112)
#         self.bn1   = model._modules['bn1']                          # output: (bs, 64, 112, 112)
        
#         self.relu    = model._modules['relu']                       # output: (bs, 64, 112, 112)
#         self.maxpool = model._modules['maxpool']                    # output: (bs, 64,  56,  56)
#         self.layer1  = model._modules['layer1']                     # output: (bs, 256, 56, 56)
#         self.layer2  = model._modules['layer2']                     # output: (bs, 512, 28, 28)
#         self.avgpool = model._modules['avgpool']                    # output: (bs, 128, 1, 1)
#         self.fc      = nn.Linear(128, 1000)                         # output: (bs, 1000)
#         del model

#         self.relu    = nn.ReLU(inplace=True)                        # output: (bs, 1000)
#         self.dense   = nn.Linear(1000, 5)                           # output: (bs, 5)
#         self.tanh = nn.Tanh()               # <- only to be applied on the first 4 elements
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)

#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#         x = self.relu(x)
#         x = self.dense(x) # the fifth component is distance; shape: (bs, 5)
#         # x[:,:4] = self.tanh(x[:,:4])
#         # x[:,4]  = self.sigmoid(x[:,4])
#         return x
