import torch
import torch.nn as nn

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # QuantStub converts tensors from floating point to quantized
        self.quant = torch.ao.quantization.QuantStub()
        self.conv = torch.nn.Conv2d(3, 512, 1)
        self.bn = torch.nn.BatchNorm2d(1)
        self.relu = torch.nn.ReLU()
        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = torch.nn.Linear(in_features=512, out_features=10, bias=True)
        # DeQuantStub converts tensors from quantized to floating point
        self.dequant = torch.ao.quantization.DeQuantStub()

    def initialize_weights(self):
       for m in self.modules():
         if isinstance(m, nn.Conv2d):
           torch.nn.init.xavier_normal(m.weight.data)
         elif isinstance(m, nn.BatchNorm2d):
           m.weight.data.fill_(1)
           m.bias.data.zero_()
         elif isinstance(m, nn.Linear):
           torch.nn.init.normal(m.weight.data, 0, 0.01)
           m.bias.data.zero_()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.dequant(x)
        return x
