import torch
import torch.nn as nn
from math import sqrt

class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        #print("Conv_ReLU_Block.size=",self.relu(self.conv(x)).size())
        return self.relu(self.conv(x))
        
        
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 18)
        self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
            #print("layers.size=",layers.size())
        return nn.Sequential(*layers)

    def forward(self, x):
        #print("x.size=",x.size())
        residual = x
        out = self.relu(self.input(x))
        #print("out1.size=",out.size())
        out = self.residual_layer(out)
        #print("out2.size=",out.size())
        out = self.output(out)
        #print("out3.size=",out.size())
        # print(out)
        # print("\n-------------\n")
        # print(residual)
        out = torch.add(out,residual)
        
        #print("out4.size=",out.size())
        return out
 