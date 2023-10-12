import numpy as np
import torch
import torch.nn as nn

class CategoricalEmbedding(nn.Module):
    def __init__(self, embedding_dim, category_num:list):
        super(CategoricalEmbedding, self).__init__()

        self.embedding = nn.ModuleList([nn.Embedding(num, embedding_dim) for num in category_num])

    def forward(self, input):

        categorical_output = []

        for idx, emb in enumerate(self.embedding):
            output = emb(input[:, :, idx:idx+1]) # (batch_size,sequence_len,1,d_model)
            categorical_output.append(output)

        return torch.cat(categorical_output, dim=2) # (batch_size,sequence_len,num_feature,d_model)


class ContinuousEmbedding(nn.Module):
    def __init__(self, embedding_dim, continuous_num):
        super(ContinuousEmbedding, self).__init__()

        self.embedding = nn.ModuleList(np.repeat(nn.Linear(1, embedding_dim), continuous_num).tolist())

    def forward(self, input):

        continuous_output = []

        for idx, emb in enumerate(self.embedding):
            output = emb(input[:, :, idx:idx+1]).unsqueeze(-2) # (batch_size,sequence_len,1,d_model)
            continuous_output.append(output)

        return torch.cat(continuous_output, dim=2) # (batch_size,sequence_len,num_feature,d_model)


class GLU(nn.Module):
    def __init__(self, d_model, output_size):
        super(GLU, self).__init__()

        self.d_model = d_model
        self.output_size = output_size

        self.sigmoid = nn.Sigmoid()
        self.linear4 = nn.Linear(self.d_model, self.output_size)
        self.linear5 = nn.Linear(self.d_model, self.output_size)

    def forward(self, input):

        out1 = self.linear4(input)
        out2 = self.linear5(input)
        output = self.sigmoid(out1) * out2

        return output
    

class GRN(nn.Module):
    def __init__(self, input_size, d_model, output_size, dropout):
        super(GRN, self).__init__()

        self.input_size = input_size
        self.d_model = d_model
        self.output_size = output_size
        self.dropout = dropout

        self.elu = nn.ELU()
        self.linear1 = nn.Linear(self.d_model, self.d_model)
        self.linear2 = nn.Linear(self.input_size, self.d_model)
        self.linear3 = nn.Linear(self.d_model, self.d_model, bias=False) # c와 내적

        if self.input_size != self.output_size:
            self.skip_layer = nn.Linear(self.input_size, self.output_size)

        self.layernorm = nn.LayerNorm(self.output_size)
        self.dropout1 = nn.Dropout(self.dropout)
        self.dropout2 = nn.Dropout(self.dropout)

        self.glu = GLU(d_model, output_size)

    def forward(self, input, c=None):

        if self.input_size != self.output_size:
            resid = self.skip_layer(input)
        else:
            resid = input

        if c is not None:
            eta2 = self.elu(self.linear2(input) + self.linear3(c))
        else:
            eta2 = self.elu(self.linear2(input))

        eta1 = self.linear1(eta2)
        eta1 = self.dropout1(eta1)
        output = self.dropout2(resid + self.glu(eta1))
        output = self.layernorm(output)

        return output


class VariableSelection(nn.Module):
    def __init__(self, d_model:int, num_inputs:int, dropout=0.1):
        super(VariableSelection, self).__init__()

        self.input_size = d_model
        self.d_model = d_model
        self.output_size = d_model
        self.num_inputs = num_inputs
        self.dropout = dropout

        self.grn_v = GRN(self.input_size*self.num_inputs, self.d_model, self.num_inputs, self.dropout)
        self.grn_ksi = nn.ModuleList(np.repeat(GRN(self.input_size, self.d_model, self.output_size, self.dropout), self.num_inputs).tolist())
        self.softmax = nn.Softmax()

    def forward(self, ksi, c=None):

        KSI = torch.flatten(ksi, -2)
        v = self.softmax(self.grn_v(KSI, c)) # (batch_size,sequence_len,num_inputs)

        ksi_set = []
        for idx, grn in enumerate(self.grn_ksi):
            ksi_tilde = grn(ksi[:, :, idx:idx+1], None) # (batch_size,sequence_len,1,d_model)
            ksi_set.append(ksi_tilde)
        ksi_tilde_set = torch.cat(ksi_set, dim=2) # (batch_size,sequence_len,num_inputs,d_model)

        output = torch.matmul(v.unsqueeze(2), ksi_tilde_set).squeeze(2)

        return output # (batch_size,sequence_len,d_model)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout):
        super(ScaledDotProductAttention, self).__init__()

        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = dropout

        self.softmax = nn.Softmax(dim=2)

    def forward(self, query, key, value, mask=None):

        attention = torch.bmm(query, key.permute(0,2,1)) # (batch,q_squence,d_model) x (batch,d_model,squence)
        # bmm : batch matrix multiplication
        scaling = torch.as_tensor(query.shape[-1], dtype=attention.dtype).sqrt()
        attention = attention/scaling # (batch,q_sequence,sequence)

        if mask is not None:
            attention = attention.masked_fill(mask, -float('inf')) # True인 위치에 -inf를 채워넣음

        attention = self.softmax(attention)

        if self.dropout is not None:
            attention = self.dropout(attention)

        output = torch.bmm(attention, value) # (batch,q_squence,squence) x (batch,squence,d_model)

        return output, attention # (batch,q_squence,d_model), (batch,q_squence,squence)


class InterpretableMultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super(InterpretableMultiheadAttention, self).__init__()
                
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_attn = self.d_model // self.num_heads
        self.d_v = self.d_attn
        self.dropout = nn.Dropout(dropout)

        self.attention = ScaledDotProductAttention(dropout)

        self.q_linear = nn.ModuleList(np.repeat(nn.Linear(self.d_model, self.d_attn), self.num_heads).tolist())
        self.k_linear = nn.ModuleList(np.repeat(nn.Linear(self.d_model, self.d_attn), self.num_heads).tolist())
        self.v_linear = nn.Linear(self.d_model, self.d_v)
        self.h_linear = nn.Linear(self.d_attn, self.d_model)

    def forward(self, query, key, value, mask=None):

        heads = []
        attentions = []
        v_w = self.v_linear(value)

        for i in range(self.num_heads):
            q_w = self.q_linear[i](query)
            k_w = self.k_linear[i](key)
            head, attention = self.attention(q_w, k_w, v_w, mask)
            head = self.dropout(head)
            heads.append(head)
            attentions.append(attention)

        heads = torch.stack(heads, dim=-1)
        attention = torch.stack(attentions, dim=-1)

        h_tilde = torch.mean(heads, dim=-1) # (batch,q_sequence,d_attn)
        output = self.h_linear(h_tilde) # (batch,q_sequence,d_model)
        output = self.dropout(output)

        return output, attention # (batch,q_sequence,d_model), (batch,q_sequence,sequence,num_heads)


class QuantileOutput(nn.Module):
    def __init__(self, d_model, quantile:list, tau):
        super(QuantileOutput, self).__init__()

        self.tau = tau
        self.q_linears = nn.ModuleList(np.repeat(nn.Linear(d_model, 1), len(quantile)).tolist())

    def forward(self, input):

        quantile_outputs = []

        for _, q_linear in enumerate(self.q_linears):

            outputs = []

            for idx in range(self.tau):

                output = q_linear(input[:,idx]) # (batch,d_model) -> (batch,1)
                outputs.append(output)

            output_stack = torch.stack(outputs, dim=1) # (batch,sequence,1)
            quantile_outputs.append(output_stack)

        quantile_outputs = torch.cat(quantile_outputs, dim=-1) # (batch,sequence,len(quantile))

        return quantile_outputs