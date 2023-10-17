import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model_sample import *


# 1.Data
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 128
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)


# 2.Model
model = M()
model.initialize_weights()
model.eval()

model.qconfig = torch.quantization.get_default_qat_qconfig('x86')  # QAT config
model_fused = torch.ao.quantization.fuse_modules(model, [['conv', 'bn', 'relu']])
model_prepared = torch.ao.quantization.prepare_qat(model_fused.train())


# 3.Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_prepared.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)


# 4.Start training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_prepared.to(device)

print("Start QAT")
num_epochs = 1
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model_prepared(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 200 == 199:
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 200:.3f}")
            running_loss = 0.0

print("QAT finished")


# 5.Convert to quantization model
model_prepared.eval()
quant_model = torch.ao.quantization.convert(model_prepared)


# 6.Convert to onnx format
input_tensor = torch.randn(1, 3, 32, 32)
onnx_path = "quantized_model.onnx"
quant_model.load_state_dict( torch.load("./output/qat.pth", map_location=torch.device('cpu')) )
torch.onnx.export(quant_model, input_tensor, onnx_path, opset_version=13, input_names = ['input'], output_names = ['output'])
print("Save onnx model")
