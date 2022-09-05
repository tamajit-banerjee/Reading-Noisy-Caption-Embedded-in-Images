import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, channels, kernel_size, stride=1):

        super().__init__()
        c_0, c_1, c_2, c_3 = channels
        kernel_size = (kernel_size, kernel_size)
        padding = (1,1)
        stride = (stride, stride)
        
        self.conv1 = nn.Conv2d(in_channels = c_0, out_channels = c_1, kernel_size=(1,1), stride=stride, padding=0)
        self.bn1 = nn.BatchNorm2d(c_1)
        self.relu = nn.ReLU(inplace=True) 
        self.conv2 = nn.Conv2d(in_channels = c_1, out_channels = c_2, kernel_size=kernel_size, stride=(1,1), padding=padding)
        self.bn2 = nn.BatchNorm2d(c_2)
        self.conv3 = nn.Conv2d(in_channels = c_2, out_channels = c_3, kernel_size=(1,1), stride=(1,1), padding=0)
        self.bn3 = nn.BatchNorm2d(c_3)
        self.conv4 = nn.Conv2d(in_channels = c_0, out_channels = c_3, kernel_size=(1,1), stride=stride, padding=0)
        self.bn4 = nn.BatchNorm2d(c_3)
        
    def forward(self,x):
        x_residual = x.clone()
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x_residual = self.conv4(x_residual)
        x_residual = self.bn4(x_residual)

        x += x_residual 
        x = self.relu(x)
        return x



class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, stride=1, padding = (1,1)):

        super().__init__()
        c_0, c_1, c_2, c_3 = channels
        kernel_size = (kernel_size, kernel_size)
        stride = (stride, stride)
        
        self.relu = nn.ReLU(inplace=True) 

        self.conv1 = nn.Conv2d(in_channels = c_0, out_channels = c_1, kernel_size=(1,1), stride=stride, padding=0)
        self.bn1 = nn.BatchNorm2d(c_1)
        self.conv2 = nn.Conv2d(in_channels = c_1, out_channels = c_2, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(c_2)
        self.conv3 = nn.Conv2d(in_channels = c_2, out_channels = c_3, kernel_size=(1,1), stride=stride, padding=0)
        self.bn3 = nn.BatchNorm2d(c_3)
        
    def forward(self, x):
        x_residual = x.clone()
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        
        x += x_residual 
        x = self.relu(x)
        return x
    


    
class BaseResNet(nn.Module):
    def __init__(self, in_channels, out_dim, stage_blocks):
        super().__init__()
        
        self.pad = nn.ZeroPad2d((1, 1, 3, 3))   

        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels=64, kernel_size=(7,7), stride = (2,2), padding=1) 
        self.batch_norm1 = nn.BatchNorm2d(64) 
        self.pool1 = nn.MaxPool2d((3,3), stride=(2,2), padding=1, dilation=1)
        
        self.stage1 = self._init_stage(
            stage_blocks[0], 
            channels_conv=[64, 64, 64, 256], 
            channels_res=[256, 64, 64, 256],
            kernel_size = 3, 
            stride = 1,
        )
        self.stage2 = self._init_stage(
            stage_blocks[1], 
            channels_conv=[256, 128, 128, 512], 
            channels_res=[512, 128, 128, 512],
            kernel_size = 3, 
            stride = 1,
        )
        self.stage3 = self._init_stage(
            stage_blocks[2], 
            channels_conv=[512, 256, 256, 1024], 
            channels_res=[1024, 256, 256, 1024],
            kernel_size = 3, 
            stride = 1,
        )
        self.stage4 = self._init_stage(
            stage_blocks[3], 
            channels_conv=[1024, 512, 512, 2048], 
            channels_res=[2048, 512, 512, 2048],
            kernel_size = 3, 
            stride = 1,
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size = (1,1))
        self.fc1 = nn.Linear(in_features=2048, out_features=out_dim, bias = True)
        
    def _init_stage(self, no_blocks, channels_conv, channels_res, kernel_size=3, stride=1):
        stage_layers = []
        stage_layers.append(ConvBlock(channels=channels_conv, kernel_size=kernel_size, stride=stride))
        for i in range(no_blocks-1):
            stage_layers.append(ResBlock(channels=channels_res, kernel_size=kernel_size))
        stage = nn.Sequential(*stage_layers)
        return stage
        
    def forward(self, x):

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.pool1(x)
        
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.adaptive_pool(x)
        x = x.view(x.size()[0], -1) 
        x = self.fc1(x)

        return x
        

class ResNet50(BaseResNet):
    def __init__(self, in_channels=3, out_dim=128):
        super().__init__(
            in_channels=in_channels,
            out_dim=out_dim,
            stage_blocks=[3,4,6,3]
        )

class ResNet101(BaseResNet):
    def __init__(self, in_channels=3, out_dim=128):
        super().__init__(
            in_channels=in_channels,
            out_dim=out_dim,
            stage_blocks=[3,4,23,3]
        )

