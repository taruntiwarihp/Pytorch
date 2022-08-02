import torch
from torch import nn
from torch.nn.modules import module
from torchvision.models import wide_resnet50_2, inception_v3, vgg16_bn, densenet121, squeezenet1_1


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    '''
    3 X 3 convolution with padding 
    '''
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, dilation=dilation, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    '''
    1 X 1 convolution
    '''
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False, stride=stride)


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        if groups != 1 or base_width != 64:
            raise NotImplementedError('Not supported')

        if dilation > 1:
            raise NotImplementedError('Not supported') 

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identify = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identify = self.downsample(x)

        out += identify
        out = self.relu(out)

        return out

class HairClassificationModelWSR(torch.nn.Module):
	def __init__(self, n_classes=2):
		super(HairClassificationModelWSR, self).__init__()

		self.base = wide_resnet50_2(pretrained=True)
		self.base.fc = torch.nn.Linear(self.base.fc.in_features, n_classes)

	def forward(self, x):
		x = self.base(x)
		return x

class HairClassificationModelMobile(torch.nn.Module):
	def __init__(self, n_classes=2):
		super(HairClassificationModelMobile, self).__init__()

		self.base = mobilenet_v3_large(pretrained=True)
		self.base.classifier[3] = torch.nn.Linear(self.base.classifier[3].in_features, n_classes)

	def forward(self, x):
		x = self.base(x)
		return x

class HairClassificationModelVGG(torch.nn.Module):
	def __init__(self, n_classes):
		super(HairClassificationModelVGG, self).__init__()

		self.base = vgg16_bn(pretrained=True)
		self.base.classifier[6] = torch.nn.Linear(self.base.classifier[6].in_features, n_classes)

	def forward(self, x):
		x = self.base(x)
		return x

class HairClassificationModelDensenet(torch.nn.Module):
	def __init__(self, n_classes):
		super(HairClassificationModelDensenet, self).__init__()

		self.base = densenet121(pretrained=True)
		self.base.classifier = torch.nn.Linear(self.base.classifier.in_features, n_classes)

	def forward(self, x):
		x = self.base(x)
		return x

class HairClassificationModelSqueezenet(torch.nn.Module):
	def __init__(self, n_classes):
		super(HairClassificationModelSqueezenet, self).__init__()

		self.base = squeezenet1_1(pretrained=True)
		self.base.classifier[1] = torch.nn.Conv2d(512, n_classes, kernel_size=(1, 1), stride=(1, 1))

	def forward(self, x):
		x = self.base(x)
		return x

class HairClassificationModelXception(torch.nn.Module):
	def __init__(self, n_classes=2):
		super(HairClassificationModelXception, self).__init__()

		self.base = inception_v3(pretrained=True)
		self.base.AuxLogits.fc = torch.nn.Linear(self.base.AuxLogits.fc.in_features, n_classes)
		self.base.fc = torch.nn.Linear(self.base.fc.in_features, n_classes)

	def forward(self, x):
		x = self.base(x)
		return x

class HairClassificationModelBase(torch.nn.Module):

	def __init__(self, n_classes=2, replace_stride_with_dilation=None, norm_layer=None):
		super(HairClassificationModelBase, self).__init__()

		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		
		self._norm_layer = norm_layer

		self.inplanes = 64
		self.dilation = 1
		self.groups = 1
		self.base_width = 64
        
		if replace_stride_with_dilation is None:
			replace_stride_with_dilation = [False, False, False]

		if len(replace_stride_with_dilation) != 3:
			raise ValueError('Not Applicable')

		self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn1 = norm_layer(self.inplanes)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

		self.layer1 = self._make_layer(BasicBlock, 64, 2)
		self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2, dilate=replace_stride_with_dilation[0])
		self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2, dilate=replace_stride_with_dilation[1])
		self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2, dilate=replace_stride_with_dilation[2])

		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc = nn.Linear(512 , n_classes)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
		norm_layer =self._norm_layer
		downsample = None

		previous_dilation = self.dilation

		if dilate:
			self.dilation *= stride
			stride = 1
        
		if stride != 1 or self.inplanes != planes :
			downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                norm_layer(planes)
            )

		layers = []

		layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
		
		self.inplanes = planes

		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
     
		return nn.Sequential(*layers)

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

		return x


# TMPDIR=/data/vincents/ pip install --cache-dir=/data/vincents/ --build /data/vincents/ tensorflow-g