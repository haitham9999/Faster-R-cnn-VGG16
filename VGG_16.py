import torch.nn.functional as F
from torch import nn 
import torchvision
import torch

class VGG16(nn.Module):
  def __init__(self, ):
    super(VGG16, self).__init__()

    self.model = nn.Sequential(*list(torchvision.models.vgg16_bn(pretrained=True).features.children())[:-7])
    for i, param in enumerate(self.model.parameters()):
        param.requires_grad = False
    

  def forward(self, x):
 
    return self.model(x)
	
	
	



	
	
	


