import torch
import torch.nn as nn

# 1. 2 Convolution[(3x3),Relu] --> TransposeConv(out_channel,out_channel//2)[kernel = 2x2 , stride =2x2]


class UpConvolution(nn.Module):
    def __init__(self,input_channel,output_channel):
        super(UpConvolution, self).__init__()
        self.conv1 = nn.Conv2d(input_channel,output_channel,(3,3))
        self.conv2 = nn.Conv2d(output_channel,output_channel,(3,3))
        self.convtranspose = nn.ConvTranspose2d(output_channel,output_channel//2,(2,2),(2,2))
        self.Relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.Relu(x)
        x = self.conv2(x)
        x = self.Relu(x)
        x = self.dropout(x)
        x = self.convtranspose(x)

        return x

if __name__=="__main__":
    up_conv = UpConvolution(512,256)
    inp = torch.rand(4,512,104,104)
    out = up_conv(inp)
    print(out.size())


