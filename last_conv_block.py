import torch
import torch.nn as nn

# 1. 2 convolution[(3x3),Relu] --> 1x1 conv block (channel equals number of classes)

class LastConvolution(nn.Module):
    def __init__(self,input_channel,output_channel,num_classes):
        super(LastConvolution, self).__init__()
        self.conv1 = nn.Conv2d(input_channel,output_channel,(3,3))
        self.conv2 = nn.Conv2d(output_channel,output_channel,(3,3))
        self.conv1d = nn.Conv1d(output_channel,num_classes,(1,1))
        self.Relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.Relu(x)
        x = self.conv2(x)
        x = self.Relu(x)
        x = self.dropout(x)
        x = self.conv1d(x)

        return x

if __name__=="__main__":
    last_conv = LastConvolution(128,64,2)
    inp = torch.rand(4,128,392,392)
    out = last_conv(inp)
    print(out.size())


