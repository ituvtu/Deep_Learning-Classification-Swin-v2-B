# model.py
import torch
import torch.nn as nn
import timm
from timm.models.swin_transformer_v2 import SwinTransformerV2

IMAGES_SIZE=(256, 256)

class SwinV2Model(nn.Module):
    def __init__(self, num_classes=31):
        super(SwinV2Model, self).__init__()
        self.model = timm.create_model('swinv2_base_window16_256', pretrained=False, num_classes=num_classes)
    def forward(self, x):
        return self.model(x)

model_eval=SwinV2Model()
model_eval.load_state_dict(torch.load('model.pt', weights_only=True))