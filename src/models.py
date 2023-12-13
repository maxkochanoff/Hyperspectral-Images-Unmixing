import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    def __init__(self, channel_number, source_number, conv3_size=11, alpha=3.5):
        super(Autoencoder, self).__init__()
        self.conv1 = nn.Conv2d(channel_number, 48, 3, padding='same', bias=False)  
        self.conv2 = nn.Conv2d(48, source_number, 1, padding='same', bias=False)
        self.conv3 = nn.Conv2d(source_number, channel_number, conv3_size, padding='same', bias=False)
        self.batch_norm1 = nn.BatchNorm2d(48)
        self.batch_norm2 = nn.BatchNorm2d(source_number)
        self.dropout2d = nn.Dropout2d(p=0.2)
        
        self.alpha = alpha

    def forward(self, x):
        # encoder
        x = F.leaky_relu(self.conv1(x))
        x = self.batch_norm1(x)
        x = self.dropout2d(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.batch_norm2(x)
        x = self.dropout2d(x)
        encode = F.softmax(self.alpha * x, dim=1)
        
        conv3_weights = self.conv3.weight.sum(dim=-1).sum(dim=-1)
        
        # decoder
        x = self.conv3(encode)

        # X, A_gt, S_gt
        return x, conv3_weights, encode
