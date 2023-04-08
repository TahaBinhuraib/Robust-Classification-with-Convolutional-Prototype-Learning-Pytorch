from Models import ModifiedResNet
import torch

a = ModifiedResNet(32, 2).to("cpu")

x = torch.randn(1, 3, 64, 64)
print(a(x)[0].shape)
print(a(x)[1].shape)
print(a(x)[2].shape)
print(a(x)[3].shape)
