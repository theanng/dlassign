from torch.jit import load
import segmentation_models_pytorch as smp
import torch
import torch.optim as optim

learning_rate = 1e-04

model = smp.UnetPlusPlus(
    encoder_name="resnet18",      
    encoder_weights="imagenet",     
    in_channels=3,                 
    classes=3                       
)
pretrained_path = "/unetmodel.pth"
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

checkpoint = torch.load(pretrained_path)

optimizer.load_state_dict(checkpoint['optimizer'])

from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in checkpoint['model'].items():
    name = k[7:] #remove `module.`
    new_state_dict[name] = v
#load params
model.load_state_dict(new_state_dict)