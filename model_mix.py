import torch
import math
import torch.nn as nn
from torch.nn import functional as F


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)

        # print(x)
        x = self.att_dropout(x)


        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)

        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)

        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)


    def forward(self, x, attn_bias=None):


        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y
        ## 实现的是transformer 和 FFN的LayerNorm 以及相关操作
        
        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y

        return x


class SGT(nn.Module):
    def __init__(
        self,
        n_layers,
        input_dim,
        hidden_dim,
        n_class,
        num_heads,
        ffn_dim,
        dropout_rate,
        attention_dropout_rate,
        args):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.n_layers = n_layers
        self.n_class = n_class
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate

        self.node_emb = nn.Linear(self.input_dim,  self.hidden_dim)
        self.hop_emb = nn.Linear(self.input_dim,  self.hidden_dim)


        
        encoders = [EncoderLayer(self.hidden_dim, self.ffn_dim, self.dropout_rate, self.attention_dropout_rate, self.num_heads)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)

        self.final_ln = nn.LayerNorm(hidden_dim)

        self.out_proj = nn.Linear(self.hidden_dim, int(self.hidden_dim / 2))



        self.Linear1 = nn.Linear(int(self.hidden_dim / 2), self.n_class)


        self.fusion_layer = nn.Linear(2 * self.hidden_dim, 1)


        self.args = args

        self.CEloss  = nn.CrossEntropyLoss()

        self.BCEloss  = nn.BCELoss()

        self.l2loss  = nn.MSELoss(reduction='mean')


        self.cls_token_node = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.cls_token_hop = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data, labels):
        
        # print(batched_data.shape)

        split_list = [self.args.sample_num_p + 1, self.args.sample_num_n + 1, self.args.sample_num_p + 1, self.args.sample_num_n + 1]
        # print(split_list)
        #正负样本的特征分离
        out = torch.split(batched_data, split_list, dim=1)


        node_p = out[0] # [b_size, (1+pk), d]
        node_n = out[1] # [b_size, (1+pk), d]

        hop_p = out[2] # [b_size, (1+pk), d]
        hop_n = out[3] # [b_size, (1+pk), d]

        node_p = self.node_emb(node_p)
        node_n = self.node_emb(node_n)

        hop_p = self.hop_emb(hop_p)
        hop_n = self.hop_emb(hop_n)

        
        if self.args.sample_num_n > 0:

            node_n = torch.split(node_n, [1, self.args.sample_num_n], dim=1)[1]
            node_n = torch.concat((self.cls_token_node.expand(batched_data.shape[0], -1, -1), node_n), dim=1)
            
            hop_n = torch.split(node_n, [1, self.args.sample_num_n], dim=1)[1]
            hop_n = torch.concat((self.cls_token_hop.expand(batched_data.shape[0], -1, -1), hop_n), dim=1)
            

        #交叉学习

        for i, l in enumerate(self.layers):

            node_n = self.layers[i](node_n)
            node_p = self.layers[i](node_p)

            hop_n = self.layers[i](hop_n)
            hop_p = self.layers[i](hop_p)

        node_p = self.final_ln(node_p)
        node_n = self.final_ln(node_n)

        hop_p = self.final_ln(hop_p)
        hop_n = self.final_ln(hop_n)

        node_out_pre = node_p[:, 0, :] - node_n[:, 0, :]
        hop_out_pre = hop_p[:, 0, :] - hop_n[:, 0, :]


        final_out = self.args.alpha*node_p[:, 0, :] + (1-self.args.alpha)*hop_p[:, 0, :]
        final_out = final_out - (self.args.alpha*node_n[:, 0, :] + (1-self.args.alpha)*hop_n[:, 0, :])







        output = self.Linear1(torch.relu(self.out_proj(final_out)))


        node_sum_p =  torch.sum(node_p[:, 1:-1, :], dim=1)/(node_p.shape[1]-1)

        hop_sum_p =  torch.sum(hop_p[:, 1:-1, :], dim=1)/(hop_p.shape[1]-1)  


        if self.args.sample_num_n > 0:

            con_loss = celoss(node_p[:, 0, :], node_sum_p, node_n[:, 1:-1, :], self.args.temp, self.args.device) + celoss(hop_p[:, 0, :] , hop_sum_p, hop_n[:, 1:-1, :], self.args.temp, self.args.device) 

        else:
            con_loss = 0



        ce_loss = self.CEloss(output, labels)
        output = F.log_softmax(output, dim=1)



        return  output, ce_loss, con_loss



def celoss(target_re, pos_re, neg_re, temperature, device):
    l_pos = torch.einsum("nc,nc->n", [target_re, pos_re]).unsqueeze(-1)
    l_neg = torch.einsum("nc,nkc->nk", [target_re, neg_re.clone().detach()])


    logits = torch.cat([l_pos, l_neg], dim=1)

    logits /= temperature

    criterion = nn.CrossEntropyLoss().cuda(device)
    
    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda(device)

    loss = criterion(logits.cuda(device), labels)

    return loss

