import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
# GPU 사용 가능 -> True, GPU 사용 불가 -> False
print(torch.cuda.is_available())