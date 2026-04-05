
#!/usr/bin/env python
# coding: utf-8

# In[1]:


# models_hybrid.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, BertModel, ViTConfig


# In[8]:


#get_ipython().run_line_magic('pip', 'install -r requirements.txt')
#get_ipython().run_line_magic('python', '-m spacy download en_core_web_sm')


# In[2]:


# Vision encoder using HuggingFace ViT
class VisionEncoderViT(nn.Module):
    def __init__(self, model_name="google/vit-base-patch16-224-in21k", out_dim=768, freeze_backbone=False):
        super().__init__()
        self.vit = ViTModel.from_pretrained(model_name)
        if freeze_backbone:
            for p in self.vit.parameters():
                p.requires_grad = False
        self.proj = nn.Linear(self.vit.config.hidden_size, out_dim)

    def forward(self, pixel_values):
        # pixel_values: [B, 3, H, W] processed by ViTFeatureExtractor
        outputs = self.vit(pixel_values=pixel_values, return_dict=True)
        # use pooled output as global representation
        pooled = outputs.pooler_output  # [B, hidden]
        last_hidden = outputs.last_hidden_state  # [B, seq_len, hidden]
        return self.proj(pooled), self.proj(last_hidden)  # [B,out_dim], [B,seq_len,out_dim]


# In[5]:


# Caption decoder: uses PyTorch TransformerDecoder but tokenizer is T5 (vocab)
class CaptionDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=768, nhead=8, num_layers=6, dim_feedforward=2048, max_len=40, pad_idx=0):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        # project vision sequence embeddings to d_model if needed (they should already be d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt_input_ids, memory, tgt_mask=None, tgt_key_padding_mask=None):
        # tgt_input_ids: [B, T] -> transformer expects [T, B, d_model]
        x = self.token_embedding(tgt_input_ids) * (self.d_model ** 0.5)  # [B, T, d]
        x = self.pos_enc(x)  # adds positional encoding; returns [B,T,d]
        x = x.permute(1, 0, 2)  # [T,B,d]
        # memory: [B, S, d] -> transformer expects [S,B,d]
        memory = memory.permute(1, 0, 2)
        if tgt_mask is None:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(x.size(0)).to(x.device)
        output = self.transformer_decoder(x, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        logits = self.fc_out(output)  # [T,B,V]
        return logits.permute(1, 0, 2)  # return [B,T,V]

    def generate_greedy(self, memory, tokenizer, max_len=30, device='cpu'):
        # memory: [B, S, d], B assumed 1 for simplicity here
        B = memory.size(0)
        assert B == 1, "greedy generation implemented for batch size 1"
        generated = [tokenizer.convert_tokens_to_ids(tokenizer.pad_token)]  # start with pad or could use <pad>
        # better approach: use tokenizer's bos token; T5 uses no bos by default; we'll rely on starting token id 0
        cur = torch.tensor([generated], dtype=torch.long).to(device)  # [1,1]
        for i in range(max_len):
            logits = self.forward(cur, memory)  # [B, T, V]
            next_token_logits = logits[:, -1, :]  # [B, V]
            next_id = torch.argmax(next_token_logits, dim=-1)  # [B]
            cur = torch.cat([cur, next_id.unsqueeze(1)], dim=1)
            if next_id.item() == tokenizer.eos_token_id:
                break
        return cur.squeeze(0).tolist()


# In[6]:


# Positional Encoding from "Attention is all you need"
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1,max_len,d]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, T, d]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


# In[7]:


# VQA hybrid head (ViT + BERT fusion)
class VQAHybrid(nn.Module):
    def __init__(self, vit_out_dim=768, bert_model_name="bert-base-uncased", hidden_dim=1024, num_answers=3000, freeze_text=False):
        super().__init__()
        self.vision = VisionEncoderViT(out_dim=vit_out_dim)
        self.bert = BertModel.from_pretrained(bert_model_name)
        if freeze_text:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.fc_img = nn.Linear(vit_out_dim, hidden_dim)
        self.fc_txt = nn.Linear(self.bert.config.hidden_size, hidden_dim)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(hidden_dim, num_answers)

    def forward(self, pixel_values, input_ids, attention_mask):
        # pixel_values: [B,3,H,W], input_ids, attention_mask: bert encodings
        img_vec, _ = self.vision(pixel_values)  # [B, vit_out_dim]
        img_proj = self.relu(self.fc_img(img_vec))
        txt_out = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        ques_vec = txt_out.pooler_output  # [B, bert_hidden]
        txt_proj = self.relu(self.fc_txt(ques_vec))
        # fusion
        fused = img_proj * txt_proj
        logits = self.classifier(fused)
        return logits


# In[ ]:




 
