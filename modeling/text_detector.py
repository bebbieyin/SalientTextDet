import torch.nn as nn

class TextDetector(nn.Module):
    def __init__(self, inner_channels):
        super(TextDetector, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, 3, bias=False, padding=1),
            nn.Conv2d(inner_channels, inner_channels // 4, 3, bias=False, padding=1),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, inner_channels // 4, 2, 2),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, 1, 2, 2),
        )

    def forward(self, x):
        return self.layers(x)
