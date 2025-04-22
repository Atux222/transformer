import torch
from torch import nn
import torch.nn.functional as F
import math

from torch import Tensor

class TokenEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size, padding_idx=0):
        super(TokenEmbedding, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)

    def forward(self, x):
        return self.embedding(x) * (self.d_model ** 0.5)  



class PositionalEmbedding(nn.Module):
    def __init__(self, d_model,max_len,device):
        super(PositionalEmbedding,self).__init__()

        self.encoding=torch.zeros(max_len,d_model,device=device)
        self.encoding.requires_grad=False

        pos=torch.arange(0,max_len,device=device)
        pos=pos.float().unsqueeze(dim=1)
        _2i=torch.arange(0,d_model,step=2,device=device).float()

        self.encoding[:,0::2]=torch.sin(pos/(10000**(_2i/d_model)))
        self.encoding[:,1::2]=torch.cos(pos/(10000**(_2i/d_model)))

    def forward(self,x):
        batch_size,seq_len=x.size()
        return self.encoding[:seq_len, :].unsqueeze(0).expand(batch_size, seq_len, -1)
    
class TransformerEmbedding(nn.Module):
    def __init__(self,vocab_size,d_model,max_len,drop_prob,device):
        super(TransformerEmbedding,self).__init__()
        self.tok_emb=TokenEmbedding(d_model,vocab_size)
        self.pos_emb=PositionalEmbedding(d_model,max_len,device)
        self.drop_out=nn.Dropout(p=drop_prob)

    def forward(self,x):
        tok_emb=self.tok_emb(x)
        pos_emb=self.pos_emb(x)
        return self.drop_out(tok_emb+pos_emb)



class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,n_head):
        super(MultiHeadAttention,self).__init__()
        self.n_head=n_head
        self.d_model=d_model
        self.d_k=self.d_model//self.n_head
        self.w_q = nn.Linear(d_model, d_model) 
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_combine = nn.Linear(d_model, d_model) 
        self.softmax=nn.Softmax(dim=-1)

    def forward (self,q,k,v,mask=None):
        batch_size, seq_len, _ = q.shape
        value_len, key_len, query_len = v.shape[1], k.shape[1], q.shape[1]

        query= self.w_q(q)
        key=self.w_k(k) 
        value=self.w_v(v)

        query = query.view(batch_size, query_len, self.n_head, self.d_k).permute(0, 2, 1, 3)  # (batch, n_head, seq_len, d_k)
        key = key.view(batch_size, key_len, self.n_head, self.d_k).permute(0, 2, 1, 3)
        value = value.view(batch_size, value_len, self.n_head, self.d_k).permute(0, 2, 1, 3)


        score=(query@key.transpose(-2,-1))/math.sqrt(self.d_k)

        if mask is not None:
            score=score.masked_fill(mask==0,-10000)

        score=self.softmax(score)
        score = score @ value
        score=score.permute(0,2,1,3).contiguous().view(batch_size, seq_len, self.d_model)
        
        out=self.w_combine(score)
        return out

        
class LayerNorm(nn.Module):
    def __init__(self,d_model,eps=1e-12):
        super(LayerNorm,self).__init__()
        self.gamma=nn.Parameter(torch.ones(d_model))
        self.beta=nn.Parameter(torch.zeros(d_model))
        self.eps=eps
    def forward(self,x):
        mean=x.mean(-1,keepdim=True)
        var=x.var(-1,unbiased=False,keepdim=True)
        out=(x-mean)/torch.sqrt(var+self.eps)
        out=self.gamma*out+self.beta
        return out
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self,d_model,hidden,dropout=0.1):
        super(PositionwiseFeedForward,self).__init__()
        self.fc1=nn.Linear(d_model,hidden)
        self.fc2=nn.Linear(hidden,d_model)
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        x=self.fc1(x)
        x=F.relu(x)
        x=self.dropout(x)
        x=self.fc2(x)
        return x
    
class  EncoderLayer(nn.Module):
     def __init__(self,d_model,ffn_hidden,n_head,dropout=0.1):
         super(EncoderLayer,self).__init__()
         
         self.attention=MultiHeadAttention(d_model,n_head)
         self.norm1=LayerNorm(d_model)
         self.dropout1=nn.Dropout(dropout)
         self.ffn=PositionwiseFeedForward(d_model,ffn_hidden,dropout)
         self.norm2=LayerNorm(d_model)
         self.dropout2=nn.Dropout(dropout)
     def forward(self,x,mask=None):
         _x=x
         x=self.attention(x,x,x,mask)
         x=self.dropout1(x)
         x=self.norm1(x+_x)
         _x=x
         x=self.ffn(x)
         x=self.dropout2(x)
         x=self.norm2(x+_x)
         return x
     
class Encoder(nn.Module):
    def __init__(self, enc_voc_size,max_len,d_model,ffn_hidden,n_head,n_layer,drop_prob,device):
        super(Encoder,self).__init__()
        self.d_model = d_model
        self.embedding=TransformerEmbedding(enc_voc_size,d_model,max_len,drop_prob,device)
        self.layers=nn.ModuleList(
            [
                EncoderLayer(d_model,ffn_hidden,n_head,drop_prob)
                for _ in range(n_layer)
            ]
        )
    def forward(self,x,s_mask):
        x=self.embedding(x)
        for layer in self.layers:
            x=layer(x,s_mask)
        return x

class DecoderLayer(nn.Module):
    def __init__(self,d_model,ffn_hidden,n_head,drop_prob):
        super(DecoderLayer,self).__init__()
        self.attention=MultiHeadAttention(d_model,n_head)
        self.norm1=LayerNorm(d_model)
        self.dropout1=nn.Dropout(drop_prob)
        self.cross_attention=MultiHeadAttention(d_model,n_head)
        self.norm2=LayerNorm(d_model)
        self.dropout2=nn.Dropout(drop_prob)
        self.ffn=PositionwiseFeedForward(d_model,ffn_hidden,drop_prob)
        self.norm3=LayerNorm(d_model)
        self.dropout3=nn.Dropout(drop_prob)
    def forward(self,dec,enc,cross_mask,t_mask):
        _x=dec
        x=self.attention(dec,dec,dec,t_mask)
        x=self.dropout1(x)
        x=self.norm1(x+_x)
        _x=x
        x=self.cross_attention(x,enc,enc,cross_mask)
        x=self.dropout2(x)
        x=self.norm2(x+_x)
        _x=x
        x=self.ffn(x)   
        x=self.dropout3(x)
        x=self.norm3(x+_x)
        return x

class Decoder(nn.Module):
    def __init__(self,dec_voc_size,max_len,d_model,ffn_hidden,n_head,n_layer,drop_prob,device):
        super(Decoder,self).__init__()
        self.d_model = d_model
        self.embedding=TransformerEmbedding(dec_voc_size,d_model,max_len,drop_prob=drop_prob,device=device)
        self.layers=nn.ModuleList(
            [
                DecoderLayer(d_model,ffn_hidden,n_head,drop_prob)
                for _ in range(n_layer)
            ]
        )
        self.fc=nn.Linear(d_model,dec_voc_size)

    def forward(self,dec,enc,cross_mask,t_mask):
        dec=self.embedding(dec)
        for layer in self.layers:
            dec=layer(dec,enc,cross_mask,t_mask)
        dec=self.fc(dec)
        return dec


class Transformer(nn.Module):
    def __init__(self,src_pad_idx,trg_pad_idx,enc_voc_size,dec_voc_size,d_model,max_len,n_head,ffn_hidden,n_layer,drop_prob,device):
        super(Transformer,self).__init__()

        self.encoder=Encoder(enc_voc_size,max_len,d_model,ffn_hidden,n_head,n_layer,drop_prob,device)
        self.decoder=Decoder(dec_voc_size,max_len,d_model,ffn_hidden,n_head,n_layer,drop_prob,device)

        if self.encoder.d_model != self.decoder.d_model:
            self.encoder_to_decoder = nn.Linear(self.encoder.d_model, self.decoder.d_model,bias=False)
        else:
            self.encoder_to_decoder = nn.Identity()

        self.src_pad_idx=src_pad_idx
        self.trg_pad_idx=trg_pad_idx
        self.device=device

    def make_pad_mask(self,q,k,pad_idx_q,pad_idx_k):
        len_q=q.size(1)
        len_k=k.size(1)
        q=q.ne(pad_idx_q).unsqueeze(1).unsqueeze(3)
        q=q.repeat(1,1,1,len_k)
        k=k.ne(pad_idx_k).unsqueeze(1).unsqueeze(2)
        k=k.repeat(1,1,len_q,1)
        mask=q&k
        return mask
    def make_causal_mask(self, q, k):
        len_q=q.size(1)
        len_k=k.size(1)
        mask = torch.tril(torch.ones((len_q, len_k), dtype=torch.bool))  # 先创建
        return mask.to(self.device)  # 再转设备

    def forward(self,src,trg):
        src_mask=self.make_pad_mask(src,src,self.src_pad_idx,self.src_pad_idx)
        trg_mask=self.make_pad_mask(trg,trg,self.trg_pad_idx,self.trg_pad_idx)&self.make_causal_mask(trg,trg)
        cross_mask = self.make_pad_mask(trg, src, self.trg_pad_idx, self.src_pad_idx)


        enc=self.encoder(src,src_mask)
        enc = self.encoder_to_decoder(enc)
        out=self.decoder(trg,enc, cross_mask,trg_mask)
        return out




    
   
