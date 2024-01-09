from args import args

MAX_LEN = 100

import torch
import torch.nn as nn
import numpy as np

device = 'cuda:0' if torch.cuda.is_available() and args.gpu else 'cpu'

class MaskedMultiheadAttention(nn.Module):
    """
    A vanilla multi-head masked attention layer with a projection at the end.
    """
    def __init__(self, mask=False):
        super(MaskedMultiheadAttention, self).__init__()
        assert args.nhid_tran % args.nhead == 0
        # mask : whether to use
        # key, query, value projections for all heads
        self.key = nn.Linear(args.nhid_tran, args.nhid_tran)
        self.query = nn.Linear(args.nhid_tran, args.nhid_tran)
        self.value = nn.Linear(args.nhid_tran, args.nhid_tran)
        # regularization
        self.attn_drop = nn.Dropout(args.attn_pdrop)
        # output projection
        self.proj = nn.Linear(args.nhid_tran, args.nhid_tran)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        if mask:
            self.register_buffer("mask", torch.tril(torch.ones(MAX_LEN, MAX_LEN)))
        self.nhead = args.nhead
        self.d_k = args.nhid_tran // args.nhead

    def forward(self, q, k, v, mask=None):
        # WRITE YOUR CODE HERE

        Q = self.query(q)
        Q_size = Q.size()
        Q = Q.reshape([Q_size[0], Q_size[1], self.nhead, -1])
        Q = torch.transpose(Q, 1, 2)

        K = self.key(k)
        K_size = K.size()
        K = K.reshape([K_size[0], K_size[1], self.nhead, -1])
        K = torch.transpose(K, 1, 2)

        V = self.value(v)
        V_size = V.size()
        V = V.reshape([V_size[0], V_size[1], self.nhead, -1])
        V = torch.transpose(V, 1, 2)

        K = torch.transpose(K, 2, 3)
        R = torch.matmul(Q, K)
        R = R/(self.d_k ** (1/2))

        #casual masking
        if hasattr(self, "mask"):
          temp_mask = self.mask[:R.size(2),:R.size(3)]
          temp_mask = temp_mask < 0.5
          R = R.masked_fill_(temp_mask, -float('Inf'))

        #Pad masking
        if mask != None:
          mask = mask < 0.5
          R = R.permute([2,1,0,3])
          R = R.masked_fill_(mask.to(device), -float('Inf'))
          R = R.permute([2,1,0,3])

        R = torch.nn.Softmax(dim=-1)(R)
        R = self.attn_drop(R)
        output = torch.matmul(R, V)

        output = output.transpose(1,2)
        output = output.reshape(output.size(0), output.size(1), -1)
        output = self.proj(output)

        assert output != None , output
        return output


class TransformerEncLayer(nn.Module):
    def __init__(self):
        super(TransformerEncLayer, self).__init__()
        self.ln1 = nn.LayerNorm(args.nhid_tran)
        self.ln2 = nn.LayerNorm(args.nhid_tran)
        self.attn = MaskedMultiheadAttention()
        self.dropout1 = nn.Dropout(args.resid_pdrop)
        self.dropout2 = nn.Dropout(args.resid_pdrop)
        self.ff = nn.Sequential(
            nn.Linear(args.nhid_tran, args.nff),
            nn.ReLU(),
            nn.Linear(args.nff, args.nhid_tran)
        )

    def forward(self, x, mask=None):
        # WRITE YOUR CODE HERE
        output = self.ln1(x)
        output = output + self.dropout1(self.attn(output, output, output, mask))
        output = self.ln2(output)
        output = output + self.dropout2(self.ff(output))

        return output
    
class PositionalEncoding(nn.Module):
    def __init__(self, max_len=4096):
        super().__init__()
        dim = args.nhid_tran
        pos = np.arange(0, max_len)[:, None]
        i = np.arange(0, dim // 2)
        denom = 10000 ** (2 * i / dim)

        pe = np.zeros([max_len, dim])
        pe[:, 0::2] = np.sin(pos / denom)
        pe[:, 1::2] = np.cos(pos / denom)
        pe = torch.from_numpy(pe).float()

        self.register_buffer('pe', pe)

    def forward(self, x):
        # DO NOT MODIFY
        return x + self.pe[:x.shape[1]]

class TransformerEncoder(nn.Module):

    def __init__(self):
        super(TransformerEncoder, self).__init__()
        # input embedding stem
        self.tok_emb = nn.Linear(5, args.nhid_tran) #ohlev encoding
        self.pos_enc = PositionalEncoding()
        self.dropout = nn.Dropout(args.embd_pdrop)
        # transformer
        self.transform = nn.ModuleList([TransformerEncLayer() for _ in range(args.nlayers_transformer)])
        # decoder head
        self.ln_f = nn.LayerNorm(args.nhid_tran)
        self.classifier_head = nn.Sequential(
           nn.Linear(args.nhid_tran, args.nhid_tran),
           nn.LeakyReLU(),
           nn.Dropout(args.embd_pdrop),
           nn.Linear(args.nhid_tran, args.nhid_tran),
           nn.LeakyReLU(),
           nn.Linear(args.nhid_tran, 3),
           nn.Softmax(dim=1),
       )


    def forward(self, x, mask=None):
        # WRITE YOUR CODE HERE
        output = self.tok_emb(x)
        output = self.pos_enc(output)
        output = self.dropout(output)

        for i in range(args.nlayers_transformer):
          output = self.transform[i](output, mask=mask)

        output = self.ln_f(output)
        output = output.mean(dim=1)
        output = self.classifier_head(output)

        return output


