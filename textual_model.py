import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from torch.nn import init
from einops import rearrange

class AttentionPool(nn.Module):
    def __init__(self, token_num:int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(token_num + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = rearrange(x,'B N C -> N B C')
        #print(x.shape)
        #x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=True
        )
        #print(x.squeeze(0).shape)
        return x.squeeze(0)


class Textual_encoder(nn.Module):
    def __init__(self, encoder_type: str):
        super(Textual_encoder, self).__init__()
        self.encoder = BertModel.from_pretrained(encoder_type)
        self.probe_linearing = nn.Linear(768,768)
        #self.attnpool = AttentionPool(64,768,8,768)
        unfreeze_layers = ['layer.8','layer.9','layer.10', 'layer.11', 'pooler']
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break
        #self.initialize_parameters()

    # def initialize_parameters(self):
    #     if self.attnpool is not None:
    #         std = self.attnpool.c_proj.in_features ** -0.5
    #         nn.init.normal_(self.attnpool.q_proj.weight, std=std)
    #         nn.init.normal_(self.attnpool.k_proj.weight, std=std)
    #         nn.init.normal_(self.attnpool.v_proj.weight, std=std)
    #         nn.init.normal_(self.attnpool.c_proj.weight, std=std)

    def get_global_embedding(self,token,mask):
        x = self.encoder(input_ids=token, attention_mask=mask)
        hidden_states = x.last_hidden_state
        #print(hidden_states.shape)
        pooler_output = torch.mean(hidden_states, dim=1)
        #pooler_output = x.pooler_output
        #pooler_output = self.probe_linearing(pooler_output)
        return pooler_output

    # def get_attnpool_global_embedding(self,token,mask):
    #     x = self.encoder(input_ids=token, attention_mask=mask)
    #     hidden_states = x.last_hidden_state
    #     #print(hidden_states.shape)
    #     #print("Using attnpool")
    #     pooler_output = self.attnpool(hidden_states)
    #     return pooler_output

    def get_local_embedding(self,token,mask):
        x = self.encoder(input_ids=token, attention_mask=mask)
        hidden_states = x.last_hidden_state
        return hidden_states

    def forward(self, token, mask):
        x = self.encoder(input_ids=token, attention_mask=mask)
        hidden_states = x.last_hidden_state
        #pooler_output = x.pooler_output
        #pooler_output = self.probe_linearing(pooler_output)
        pooler_output = torch.mean(hidden_states, dim=1)
        #pooler_output = self.attnpool(hidden_states)
        return pooler_output, hidden_states


class Text_decoder(nn.Module):
    def __init__(self, txt_dim:int, img_dim:int):
        super().__init__()
        self.fc1 = nn.Linear(img_dim + txt_dim, txt_dim,bias=True)
        self.fc2 = nn.Linear(txt_dim, txt_dim, bias=True)
        init.kaiming_normal_(self.fc1.weight, mode='fan_out')
        init.kaiming_normal_(self.fc2.weight, mode='fan_out')
    def forward(self, cat_embedding):
        x = self.fc1(cat_embedding)
        x = F.relu(x)
        x = self.fc2(x)
        return x


