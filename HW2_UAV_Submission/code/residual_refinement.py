import torch
from torch import nn
from torch.nn import functional as F


class ResidualRefinement(nn.Module):
    def __init__(self, semantic_channels, classes):
        super().__init__()
        self.spatial = nn.Sequential(nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU())
        self.semantic = nn.Conv2d(semantic_channels, 64, 1)
        self.refine = nn.Sequential(nn.Conv2d(96, 32, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.head = nn.Conv2d(32, classes, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, image, features):
        spatial = self.spatial(image)
        semantic = F.interpolate(self.semantic(features), size=spatial.shape[-2:],
                                 mode="bilinear", align_corners=False)
        residual = self.head(self.refine(torch.cat((semantic, spatial), dim=1)))
        return F.interpolate(residual, size=image.shape[-2:], mode="bilinear", align_corners=False)


@torch.inference_mode()
def verify_initialization(model, original):
    model.eval()
    original.eval()
    device = next(model.parameters()).device
    for h, w in ((512,512),(480,640),(576,768),(576,576),(640,640),(129,193)):
        image = torch.linspace(-1, 1, 3*h*w, device=device).reshape(1,3,h,w)
        torch.testing.assert_close(model(image), original(image), rtol=0, atol=0)
    print("Initial logits match exactly at all five M7 sizes and an odd input size.")
