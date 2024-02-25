import torch

BATCH_SIZE = 10
PATCH_SIZE = 40
PATCH_NUMBER = 50
NUMBER_IMAGES = 70
NOISE_STD = 0.03

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
