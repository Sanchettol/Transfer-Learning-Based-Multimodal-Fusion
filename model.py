import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertTokenizer, BertModel

device = torch.device("cpu")

# Load BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased')

class MultimodalModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.image_model = models.resnet18(pretrained=True)
        self.image_model.fc = nn.Identity()
        
        self.text_model = bert
        
        self.fc = nn.Sequential(
            nn.Linear(512 + 768, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, image, input_ids, attention_mask):
        img_feat = self.image_model(image)
        text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = text_output.pooler_output
        
        combined = torch.cat((img_feat, text_feat), dim=1)
        return self.fc(combined)

# Load model
model = MultimodalModel()
model.load_state_dict(torch.load("multimodal_model.pth", map_location=device))
model.eval()