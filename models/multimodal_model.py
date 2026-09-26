# models/multimodal_model.py

import torch
import torch.nn as nn
from transformers import RobertaModel
from torchvision.models import resnet50, ResNet50_Weights

class MultimodalModel(nn.Module):
    def __init__(self):
        super(MultimodalModel, self).__init__()

        self.roberta = RobertaModel.from_pretrained('roberta-base')

        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(768 + 2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )

    def forward(self, input_ids, attention_mask, image):
        text_feat = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).pooler_output

        img_feat = self.resnet(image)

        combined = torch.cat((text_feat, img_feat), dim=1)
        return self.classifier(combined)
