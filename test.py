import torch
print(torch.cuda.is_available())   # True
print(torch.cuda.get_device_name(0))  # NVIDIA GeForce RTX 4060   