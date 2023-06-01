import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from torch.nn import init
from einops import rearrange

class Textual_encoder(nn.Module):
    def __init__(self, encoder_type: str):
        super(Textual_encoder, self).__init__()
        self.encoder = BertModel.from_pretrained(encoder_type)
        unfreeze_layers = ['layer.8','layer.9','layer.10', 'layer.11', 'pooler']
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break
                    
    def get_global_embedding(self,token,mask):
        x = self.encoder(input_ids=token, attention_mask=mask)
        pooler_output = x.pooler_output
        return pooler_output

    def get_local_embedding(self,token,mask):
        x = self.encoder(input_ids=token, attention_mask=mask)
        hidden_states = x.last_hidden_state
        return hidden_states

    def forward(self, token, mask):
        x = self.encoder(input_ids=token, attention_mask=mask)
        hidden_states = x.last_hidden_state
        pooler_output = x.pooler_output
        return pooler_output, hidden_states


