import torch
from torch import nn
from torch.nn import functional as F

class VGGSiameseNet(nn.Module):
    def __init__(self):
        super(VGGSiameseNet, self).__init__()
        self.conv11 = nn.Conv2d(1, 64, 3) 
        self.conv12 = nn.Conv2d(64, 64, 3)  
        self.conv21 = nn.Conv2d(64, 128, 3)
        self.conv22 = nn.Conv2d(128, 128, 3)
        self.conv31 = nn.Conv2d(128, 256, 3) 
        self.conv32 = nn.Conv2d(256, 256, 3)  
        self.conv33 = nn.Conv2d(256, 256, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 8 * 8, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fcOut = nn.Linear(4096, 1)
        self.sigmoid = nn.Sigmoid()
    
    def convs(self, x):
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.max_pool2d(x, (2,2))
        x = F.relu(self.conv21(x))
        x = F.relu(self.conv22(x))
        x = F.max_pool2d(x, (2,2))
        x = F.relu(self.conv31(x))
        x = F.relu(self.conv32(x))
        x = F.relu(self.conv33(x))
        x = F.max_pool2d(x, (2,2))
        return x

    def forward(self, x1, x2):
        x1 = self.convs(x1)
        x1 = x1.view(-1, 256 * 8 * 8)
        x1 = self.fc1(x1)
        x1 = self.sigmoid(self.fc2(x1))
        x2 = self.convs(x2)
        x2 = x2.view(-1, 256 * 8 * 8)
        x2 = self.fc1(x2)
        x2 = self.sigmoid(self.fc2(x2))
        x = torch.abs(x1 - x2)
        x = self.fcOut(x)
        return x

class OmniglotModel(nn.Module):
    def __init__(self):
        super(OmniglotModel, self).__init__()
        
        # Koch et al.
        # Conv2d(input_channels, output_channels, kernel_size)
        self.conv1 = nn.Conv2d(1, 64, 10) 
        self.conv2 = nn.Conv2d(64, 128, 7)  
        self.conv3 = nn.Conv2d(128, 128, 4)
        self.conv4 = nn.Conv2d(128, 256, 4)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fcOut = nn.Linear(4096, 1)
        self.sigmoid = nn.Sigmoid()
    
    def convs(self, x):

        # Koch et al.
        # out_dim = in_dim - kernel_size + 1  
        #1, 105, 105
        x = F.relu(self.bn1(self.conv1(x)))
        # 64, 96, 96
        x = F.max_pool2d(x, (2,2))
        # 64, 48, 48
        x = F.relu(self.bn2(self.conv2(x)))
        # 128, 42, 42
        x = F.max_pool2d(x, (2,2))
        # 128, 21, 21
        x = F.relu(self.bn3(self.conv3(x)))
        # 128, 18, 18
        x = F.max_pool2d(x, (2,2))
        # 128, 9, 9
        x = F.relu(self.bn4(self.conv4(x)))
        # 256, 6, 6
        return x


    def forward(self, x1, x2):
        x1 = self.convs(x1)

        # Koch et al.
        x1 = x1.view(-1, 256 * 6 * 6)
        x1 = self.sigmoid(self.fc1(x1))
        
        x2 = self.convs(x2)

        # Koch et al.
        x2 = x2.view(-1, 256 * 6 * 6)
        x2 = self.sigmoid(self.fc1(x2))


        x = torch.abs(x1 - x2)
        x = self.fcOut(x)
        return x

class EyebrowModel(nn.Module):

    def __init__(self):
        super(EyebrowModel, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1)
        # self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        # self.bn4 = nn.BatchNorm2d(256)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 4 * 8, 4096)
        self.fcOut = nn.Linear(4096, 1)
        self.sigmoid = nn.Sigmoid()

    def convs(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, (2,2))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, (2,2))
        x = F.relu(self.bn3(self.conv3(x)))
        # # x = F.max_pool2d(x, (2,2))
        # x = F.relu(self.bn4(self.conv4(x)))
        return x

    def forward(self, x1, x2):
        x1 = self.convs(x1)

        x1 = x1.view(-1, 128 * 4 * 8)
        x1 = self.sigmoid(self.fc1(x1))
        
        x2 = self.convs(x2)

        # Koch et al.
        x2 = x2.view(-1, 128 *4 * 8)
        x2 = self.sigmoid(self.fc1(x2))


        x = torch.abs(x1 - x2)
        x = self.fcOut(x)
        return x


# conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, dilation=1)
# conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, dilation=1)
# conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1)
# conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, dilation=1)

# fc1 = nn.Linear(256 * 16 * 30, 4096)

# a = torch.randn(4, 1, 16, 32)

# b = F.relu(conv1(a))
# b = F.max_pool2d(b, (2,2))

# print(b.shape)

# b = F.relu(conv2(b))
# b = F.max_pool2d(b, (2,2))

# print(b.shape)

# b = F.relu(conv3(b))
# b = F.max_pool2d(b, (2,2))

# print(b.shape)

# b = conv4(conv3(conv2(conv1(a))))

# b = b.view(-1, 256 * 16 * 30)

# b = fc1(b)

# print(b.shape)

# model = EyebrowModel()

# out = model(a, a)