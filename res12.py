"""
code adapted from https://github.com/kjunelee/MetaOptNet/blob/master/models/ResNet12_embedding.py
"""


import torch.nn as nn
import torch
import torch.nn.functional as F
from dropblock import DropBlock

# This ResNet network was designed following the practice of the following papers:
# TADAM: Task dependent adaptive metric for improved few-shot learning (Oreshkin et al., in NIPS 2018) and
# A Simple Neural Attentive Meta-Learner (Mishra et al., in ICLR 2018).

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False, block_size=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)

    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)
        
        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20*2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size**2 * feat_size**2 / (feat_size - self.block_size + 1)**2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out


class ResNet(nn.Module):

    def __init__(self, block, keep_prob=1.0, avg_pool=False, drop_rate=0.1, dropblock_size=5):
        self.inplanes = 3
        super(ResNet, self).__init__()

        self.layer1 = self._make_layer(block, 64, stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 160, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 320, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        self.layer4 = self._make_layer(block, 640, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        if avg_pool:
            # FIXME original stride is 1
            self.avgpool = nn.MaxPool2d(5) # nn.AvgPool2d(5, stride=2)
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, output_layer=True):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.keep_avg_pool:
            x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def resnet12(keep_prob=1.0, avg_pool=False, **kwargs):
    """Constructs a ResNet-12 embedding model (i.e. without output layer).
    """
    model = ResNet(BasicBlock, keep_prob=keep_prob, avg_pool=avg_pool, **kwargs)
    return model


class ResNetWrapper(nn.Module):
    # just a wrapper applying an output layer to the res12 features
    def __init__(self, k_way, block, keep_prob=1.0, avg_pool=False, drop_rate=0.1, dropblock_size=5, interm_layer=False):
        super(ResNetWrapper, self).__init__()
        drop_rate=0.0 # turn off block drop for stage 2
        self.res12 = ResNet(block, keep_prob=keep_prob, avg_pool=avg_pool, drop_rate=drop_rate, dropblock_size=dropblock_size)
        
        if interm_layer:
            self.interm_layer = nn.Linear(16000, 1024)
        
        self.output_layer = nn.Linear(1024 if interm_layer else 16000, k_way)
        self.dropout = nn.Dropout(p=1.-keep_prob)
        self.final_layer_size = 1024 if interm_layer else 16000
    
    def forward(self, x, output_layer=True):
        self.features = self.res12(x)
        
        if hasattr(self, 'interm_layer'):
            self.features = torch.relu(self.interm_layer(self.dropout(self.features)))

        if output_layer:
          return self.output_layer(F.dropout(self.features, self.dropout.p))
        else:
          return None


def _res12(k_way, dropout, no_relu, keep_prob=1.0, avg_pool=False, **kwargs):
    """Constructs a ResNet-12 model with integrated output layer
    """
    model = ResNetWrapper(k_way, BasicBlock, keep_prob=keep_prob-dropout, avg_pool=avg_pool, **kwargs)
    return model



if False:
    """
    code adapted from https://github.com/blue-blue272/fewshot-CAN
    """

    import math
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def conv3x3(in_planes, out_planes, stride=1):
        """3x3 convolution with padding"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, bias=False)


    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, inplanes, planes, kernel=3, stride=1, downsample=None, no_relu=False):
            super(BasicBlock, self).__init__()
            if kernel == 1:
                self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
            elif kernel == 3:
                self.conv1 = conv3x3(inplanes, planes)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            if kernel == 1:
                self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
            elif kernel == 3:
                self.conv3 = conv3x3(planes, planes)
            self.bn3 = nn.BatchNorm2d(planes)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample
            self.stride = stride
            self.no_relu = no_relu

        def forward(self, x):
            residual = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual
            
            if not self.no_relu:
                out = self.relu(out)

            return out


    class Bottleneck(nn.Module):
        expansion = 4

        def __init__(self, inplanes, planes, kernel=1, stride=1, downsample=None):
            super(Bottleneck, self).__init__()
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes * 4)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample
            self.stride = stride

        def forward(self, x):
            residual = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual
            out = self.relu(out)

            return out


    class ResNet(nn.Module):

        def __init__(self, block, layers, kernel=3, k_way=5, dropout=0.5, no_relu=False):
            self.inplanes = 64
            self.kernel = kernel
            super(ResNet, self).__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)

            self.layer1 = self._make_layer(block, 64, layers[0], stride=2) 
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, no_relu=no_relu)

            self.nFeat = 512 * block.expansion
            self.dropout = nn.Dropout(p=dropout)
            self.final_layer_size = self.nFeat * 6 * 6
            self.logits = nn.Linear(self.final_layer_size, k_way)

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

        def _make_layer(self, block, planes, blocks, stride=1, no_relu=False):
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )

            layers = []
            layers.append(block(self.inplanes, planes, self.kernel, stride, downsample, no_relu))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, self.kernel))

            return nn.Sequential(*layers)

        def forward(self, x, output_layer=True):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = x.view(-1, self.nFeat * 6 * 6)
            self.features = x

            if output_layer:
              x = self.dropout(x)

              return self.logits(x)
            else:
              return None


    def res12(k_way, dropout, no_relu):
        return ResNet(BasicBlock, [1,1,1,1], kernel=3, k_way=k_way, dropout=dropout, no_relu=no_relu)
        

