# -*- encoding: utf-8 -*-
'''
@Time    :   2022/06/10 15:51:44
@Author  :   Chu Xiaokai
@Contact :   xiaokaichu@gmail.com
'''
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from utils import *
from torch.optim.lr_scheduler import LambdaLR
from args import config

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step >= num_training_steps: # end_learning_rate=8e-8
            return 0.04
        else:
            return float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 513):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return self.dropout(self.pe[:x.size(0)])

class TransformerModel(nn.Module):

    def __init__(self, ntoken, hidden, nhead, nlayers, dropout, mode='finetune'):
        super().__init__()
        print('Transformer is used for {}'.format(mode))
        self.pos_encoder = PositionalEncoding(hidden, dropout)
        encoder_layers = TransformerEncoderLayer(hidden, nhead, hidden, dropout, activation='gelu')
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.token_encoder = nn.Embedding(ntoken, hidden)
        self.segment_encoder = nn.Embedding(2, hidden)
        self.norm_layer = nn.LayerNorm(hidden)
        self.hidden = hidden
        self.mode = mode

        self.dropout = nn.Dropout(dropout)

        if mode == 'pretrain':
            self.to_logics = nn.Linear(hidden, ntoken)
            self.decoder = nn.Linear(hidden, 1)
        elif mode == 'finetune':
            self.act = nn.ELU(alpha=1.0)
            self.fc1 = nn.Linear(hidden, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 128)
            self.fc4 = nn.Linear(128, 1)
        
    def forward(self, src, src_segment, src_padding_mask=None, mlm_label=None):
        src = src.t().contiguous().cuda()
        src_segment = src_segment.t().contiguous().cuda()
        src_padding_mask = src_padding_mask.cuda()

        # transformer input 
        pos_emb = self.pos_encoder(src)  # get position embedding
        token_emb = self.token_encoder(src) # get token embedding
        seg_emb = self.segment_encoder(src_segment)  # get position embedding
        x = token_emb + pos_emb + seg_emb
        x = self.norm_layer(x)
        x = self.dropout(x)

        output = self.transformer_encoder(
            src=x, \
            mask=None,
            src_key_padding_mask=src_padding_mask,
        )

        X =  output[0, :, :]  # [seqlen, bs, 1]
        X = self.dropout(X)

        if self.mode == 'pretrain':  # for train
            scores = self.decoder(X)
            scores = torch.squeeze(scores, dim=-1)
            if self.training:
                mlm_label = mlm_label.cuda()
                output = output.transpose(0, 1) 
                logits = self.to_logics(output)  # shape = [bs, seq_len, num_tokens]
                mlm_loss = F.cross_entropy(logits.transpose(1,2), # shape=[bs, num_class, seq_len]\
                                            mlm_label,\
                                            ignore_index=config._PAD_  # _pad
                        ) 
                return scores, mlm_loss
            else:  
                return scores

        elif self.mode == 'finetune':
            h1 = self.act(self.fc1(X))
            h1 = self.dropout(h1)
            h2 = self.act(self.fc2(h1))
            h2 = self.dropout(h2)
            h3 = self.act(self.fc3(h2))
            h3 = self.dropout(h3)
            scores = self.fc4(h3)
            scores = torch.squeeze(scores, dim=-1)
            return scores
