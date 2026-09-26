# models/text_model.py

import torch.nn as nn
from transformers import RobertaModel

class TextModel(nn.Module):
    def __init__(self):
        super(TextModel, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.fc = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        return self.fc(pooled)
