# src/models/vit_bilstm_ctc.py (Fixed for Sequence Output)

import torch
import torch.nn as nn
import timm

class ViTBILSTMCTC(nn.Module):
    """
    Vision Transformer (ViT) + BiLSTM + CTC Loss for Handwriting OCR.
    """
    def __init__(self, num_classes: int, vit_model: str = 'vit_small_patch16_224'):
        super().__init__()
        
        self.vit = timm.create_model(
            vit_model, 
            pretrained=True, 
            num_classes=0, 
            global_pool='' 
        )

        vit_feature_dim = self.vit.embed_dim
        self.bilstm = nn.LSTM(
            input_size=vit_feature_dim,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        # 3. Fully Connected (FC) Classifier
        self.classifier = nn.Linear(256 * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.vit(x)
        lstm_out, _ = self.bilstm(features)
        logits = self.classifier(lstm_out)
        return logits.permute(1, 0, 2)
    
    