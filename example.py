from pyinterpx.Interpolation import interp
import torch


Interp = interp(6,4)
x = torch.rand(2, 25, 10, 10, 10)
Interp(x)
