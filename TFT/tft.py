import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import random
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import FinanceDataReader as fdr
import datetime
from dart_fss.auth import get_api_key
from dart_fss.fs import extract
import dart_fss as dart


# # Open DART API KEY 설정
# api_key='b9280e78c5a05c6491cd9e9cf1e5c5db596a73b4'
# dart.set_api_key(api_key=api_key)

# # 모든 상장된 기업 리스트 불러오기
# corp_list = dart.get_corp_list()

# # 삼성전자
# samsung = corp_list.find_by_corp_name(corp_name='00126380')

# # 2012년 01월 01일 부터 개별재무제표 검색
# fs = dart.fs.extract(corp_code='00126380', bgn_de='20200101', separate=True)
# fs_all = dart.fs.extract(corp_code='00126380', bgn_de='20100101', separate=True, report_tp=['annual', 'half', 'quarter'])


# # 재무상태표 df 생성
# bs_df = fs_all['bs'].iloc[:,7:]
# bs_df.columns = bs_df.columns.get_level_values(0)
# bs_index = fs_all['bs'].iloc[:,1:2]
# bs_index.columns = bs_index.columns.get_level_values(1)

# bs_df = pd.concat([bs_index, bs_df], axis=1)
# bs_df.to_csv('bs.csv', index=False, encoding='cp949')

# # 손익계산서 df 생성
# is_df = fs_all['is'].iloc[:,7:]
# is_df.columns = is_df.columns.get_level_values(0)
# is_index = fs_all['is'].iloc[:,1:2]
# is_index.columns = is_index.columns.get_level_values(1)

# is_df = pd.concat([is_index, is_df], axis=1)
# is_df.to_csv('is.csv', index=False, encoding='cp949')


# bs : 재무상태표
# is : 손익계산서
# cis : 포괄손익계산서
# cf : 현금흐름표

bs_df = pd.read_csv('bs.csv', encoding='cp949')
is_df = pd.read_csv('is.csv', encoding='cp949')
is_df.transpose()


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if torch.cuda.is_available():
        device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

seed_everything(42)

# # preprocessing
# def fdr_data(code, start, end):
    
#     df = fdr.DataReader(code, start, end)
    
#     return df

# def preprocessing(data, input_window, output_window):

#     data['date'] = data.index
#     data['day'] = data.date.dt.day
#     data['month'] = df.date.dt.month
#     data['dayofweek'] = df.date.dt.day_of_week
#     data.drop(columns='date', inplace=True)

#     train_set, test_set = train_test_split(data, test_size=output_window, shuffle=False)
#     train_set, valid_set = train_test_split(train_set, test_size=output_window, shuffle=False)

#     valid_set = pd.concat([train_set[-input_window:], valid_set])
#     test_set = pd.concat([valid_set[-input_window:], test_set])

#     return train_set, valid_set, test_set

# class WindowDataset(Dataset):
#     def __init__(self, data, input_window, output_window, input_size, stride=1):
        
#         L = data.shape[0]
#         num_samples = (L - input_window - output_window) // stride + 1
#         data_tensor = torch.FloatTensor(data.to_numpy())

#         X = torch.zeros(num_samples, input_window, input_size)
#         y = torch.zeros(num_samples, output_window, input_size)

#         for i in range(num_samples):
            
#             X[i,:] = data_tensor[i*stride : i*stride+input_window] # (encoder_len, num_feature)
#             y[i,:] = data_tensor[i*stride+input_window : i*stride+input_window+output_window] # (decoder_len, num_feature)

#         self.x = X
#         self.y = y
#         self.len = len(X)

#     def __getitem__(self, idx):
#         return self.x[idx], self.y[idx]

#     def __len__(self):
#         return self.len
    
# code = '005930'
# start = '2010'
# end = '2023'
# input_window = 60
# output_window = 5
# batch_size = 20

# df = fdr_data(code, start, end)    
# train_set, valid_set, test_set = preprocessing(df, input_window, output_window)

# input_size = train_set.shape[1]
# batch_size = batch_size
# output_size = input_size

# train_set = WindowDataset(train_set, input_window, output_window, input_size)
# valid_set = WindowDataset(valid_set, input_window, output_window, input_size)
# test_set = WindowDataset(test_set, input_window, output_window, input_size)

# train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
# valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False)
# test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

# train_input = next(iter(train_loader))[0]
# train_target = next(iter(train_loader))[1]

# train_input.shape # (20,60,9)
# train_target.shape # (20,5,9)

df = pd.read_csv('power/train.csv')
building = pd.read_csv('power/building_info.csv')
df = pd.merge(df, building, on='건물번호')
df = df[df['건물번호'] == 1]

def load_data():

    df = pd.read_csv('power/train.csv')
    building = pd.read_csv('power/building_info.csv')
    df = pd.merge(df, building, on='건물번호')
    df = df[df['건물번호'] == 1]

    return df

def fill_missing_with_avg(df, columns):
    for i in range(len(df)):
        if pd.isna(df.loc[i, columns]):
            
            prev_value_sum = df.loc[i-4:i-1, columns].sum()
            next_value_sum = df.loc[i+1:i+4, columns].sum()
            avg_value = (prev_value_sum + next_value_sum) / 8

            df.loc[i, columns] = avg_value
    return df

# def sin_transform(values):
#     return np.sin(2 * np.pi * values)

# def cos_transform(values):
#     return np.cos(2 * np.pi * values)

def CDH(df, num_building):
    df_ = df.copy()
    cdhs = np.array([])
    for num in range(1, num_building+1, 1):
        cdh = []
        cdh_df = df_[df_['building_num'] == num_building]
        cdh_temp = cdh_df['temperature'].values # Series로도 돌릴 수 있지만 array로 돌리는게 속도가 훨씬 빠름
        for i in range(len(cdh_temp)):
            if i < 11:
                cdh.append(np.sum(cdh_temp[:(i+1)] - 26))
            else:
                cdh.append(np.sum(cdh_temp[(i-11):(i+1)] - 26))
        
        cdhs = np.concatenate([cdhs, cdh])
    
    return cdhs

def split_data(data):

    train_set, test_set = train_test_split(data, test_size=24*7, shuffle=False)

    return train_set, test_set

def preprocess(train_set):

    train_set = train_set.drop(columns=['num_date_time', '일조(hr)', '일사(MJ/m2)', '건물유형', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)'])
    train_set.columns = ['building_num', 'date', 'temperature', 'precipitation', 'windspeed', 'humidity', 'power_consumption', 'total_area', 'cooling_area']

    train_set['precipitation'].fillna(0, inplace=True)
    train_set = fill_missing_with_avg(train_set, 'windspeed')
    train_set = fill_missing_with_avg(train_set, 'humidity')

    train_set['date'] = pd.to_datetime(train_set['date'], format='%Y%m%d %H')

    # train_set['month'] = train_set.date.dt.month
    train_set['day'] = train_set.date.dt.day
    train_set['weekday'] = train_set.date.dt.weekday
    train_set['hour'] = train_set.date.dt.hour
    # train_set['date'] = train_set.date.dt.date

    # day_periodic = (train_set['day'] - 1) / 30
    # weekday_periodic = train_set['weekday'] / 6
    # hour_periodic = train_set['hour'] / 23

    # train_set['sin_weekday'] = sin_transform(weekday_periodic)
    # train_set['cos_weekday'] = cos_transform(weekday_periodic)
    # train_set['sin_hour'] = sin_transform(hour_periodic)
    # train_set['cos_hour'] = cos_transform(hour_periodic)

    # month_dummy = pd.get_dummies(train_set['month']).rename(columns={6:'month_6', 7:'month_7', 8:'month_8'})
    # train_set = pd.concat([train_set, month_dummy[['month_6', 'month_7']]], axis=1)

    train_set['holiday'] = train_set.apply(lambda x : 0 if x['day']<5 else 1, axis = 1)
    train_set.loc[(train_set.date == datetime.date(2022, 6, 6))&(train_set.date == datetime.date(2022, 8, 15)), 'holiday'] = 1

    train_set['DI'] = 9/5*train_set['temperature'] - 0.55*(1 - train_set['humidity']/100) * (9/5*train_set['humidity'] - 26) + 32

    train_set['CDH'] = CDH(train_set, 1)

    train_set = train_set.drop(columns=['date'])
    train_set = pd.concat([train_set.iloc[:,0:5], train_set.iloc[:,6:], train_set.iloc[:,5:6]], axis=1)

    train_set = train_set[[
        'building_num', 'total_area', 'cooling_area', # static variable
        'day', 'weekday', 'hour', 'holiday', # future variable
        'temperature', 'precipitation', 'windspeed', 'humidity', 'DI', 'CDH', # observed variable
        'power_consumption' # target
        ]]

    train_set, test_set = split_data(train_set)

    train_set, valid_set = split_data(train_set)

    return train_set, valid_set, test_set, train_set.columns

class WindowDataset(Dataset):
    def __init__(self, data, input_window, output_window, input_size, stride=1):
        
        L = data.shape[0]
        self.seq_len = input_window + output_window
        num_samples = (L - self.seq_len) // stride + 1
        data_tensor = torch.tensor(data.to_numpy())

        X = torch.zeros(num_samples, input_window, input_size)
        y = torch.zeros(num_samples, output_window, input_size)

        for i in range(num_samples):
            
            X[i,:] = data_tensor[i*stride : i*stride+input_window]
            y[i,:] = data_tensor[i*stride+input_window : i*stride+self.seq_len]

        self.x = X
        self.y = y
        self.len = len(X)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.len

def loader(train_set, valid_set, test_set, batch_size):

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    return train_loader, valid_loader, test_loader

df = load_data()
train_set, valid_set, test_set, columns = preprocess(df)

train_set
valid_set
test_set

test_true = test_set.iloc[:,-1].reset_index(drop=True)

valid_set = pd.concat([train_set[-24:], valid_set]).reset_index(drop=True)
test_set = pd.concat([valid_set[-24:], test_set]).reset_index(drop=True)
# test_set.iloc[24:,-1] = 0

# scaler = MinMaxScaler()
# train_scaled = scaler.fit_transform(train_set)
# valid_scaled = scaler.transform(valid_set)
# test_scaled = scaler.transform(test_set)

input_size = train_set.shape[1]
input_window = 24 # 1 day
output_window = 5 # 5 hour
batch_size = 64

tr_set = WindowDataset(train_set, input_window, output_window, input_size)
va_set = WindowDataset(valid_set, input_window, output_window, input_size)
te_set = WindowDataset(test_set, input_window, output_window, input_size)

train_loader, valid_loader, test_loader = loader(tr_set, va_set, te_set, batch_size=batch_size)
train_input = next(iter(train_loader))[0]
train_target = next(iter(train_loader))[1]

# layer
class CategoricalEmbedding(nn.Module):
    def __init__(self, embedding_dim, category_num):
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

columns

d_model = 40 # hidden
category_num = [2,32,7,24,2]
torch.cat([train_input[:,:,0:1], train_input[:,:,3:7]], dim=-1)
torch.cat([train_input[:,:,1:3], train_input[:,:,7:]], dim=-1)

category_input = torch.tensor(torch.cat([train_input[:,:,0:1], train_input[:,:,3:7]], dim=-1), dtype=int)
continuous_input = torch.cat([train_input[:,:,1:3], train_input[:,:,7:]], dim=-1)

cat_emb = CategoricalEmbedding(d_model, category_num)
con_emb = ContinuousEmbedding(d_model, continuous_input.shape[-1])

category_input = cat_emb(category_input)
continuous_input = con_emb(continuous_input)

category_input.shape
continuous_input.shape

emb_input = torch.cat([category_input, continuous_input], dim=2)
emb_input.shape


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
    def __init__(self, d_model, num_inputs, dropout=0.1):
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
        ksi_tilde_set = torch.cat(ksi_set, dim=2)

        output = torch.matmul(v.unsqueeze(2), ksi_tilde_set).squeeze()

        return output

ksi = emb_input
varselec = VariableSelection(d_model, ksi.shape[-2])
out = varselec(ksi)
out.shape

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout):
        super(ScaledDotProductAttention, self).__init__()

        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = dropout

        self.softmax = nn.Softmax()

    def forward(self, query, key, value, mask=None):

        attention = torch.bmm(query, key.permute(0,2,1)) # (batch,squence,d_model) x (batch,d_model,squence)
        # bmm : batch matrix multiplication
        scaling = torch.as_tensor(query.shape[-1], dtype=attention.dtype).sqrt()
        attention = attention/scaling

        if mask is not None:
            attention = attention.masked_fill(mask, -float('inf')) # True인 위치에 -inf를 채워넣음

        attention = self.softmax(attention)

        if self.dropout is not None:
            attention = self.dropout(attention)

        output = torch.bmm(attention, value) # (batch,squence,squence) x (batch,squence,d_model)

        return output, attention # (batch,squence,d_model), (batch,squence,squence)

sequence = 5
attn_mask = torch.ones(sequence, sequence, dtype=torch.bool, device=device).triu(diagonal=1)
attn_mask

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

        h_tilde = torch.mean(heads, dim=-1)
        output = self.h_linear(h_tilde)
        output = self.dropout(output)

        return output, attention

num_heads=8
dropout=0.1
imha = InterpretableMultiheadAttention(d_model, num_heads, dropout)
output, attention = imha(out, out, out)
output.shape
attention

class QuantileOutput(nn.Module):
    def __init__(self, d_model, quantile, tau):
        super(QuantileOutput, self).__init__()

        self.tau = tau
        self.q_linears = nn.ModuleList(np.repeat(nn.Linear(d_model, 1), len(quantile)).tolist())

    def forward(self, input):

        quantile_outputs = []

        for _, q_linear in enumerate(self.q_linears):

            outputs = []

            for idx in range(self.tau):

                output = q_linear(input[:,idx]) # (batch,1)
                outputs.append(output)

            output_stack = torch.stack(outputs, dim=1) # (batch,sequence,1)
            quantile_outputs.append(output_stack)

        quantile_outputs = torch.cat(quantile_outputs, dim=-1) # (batch,sequence,len(quantile))

        return quantile_outputs


class TemporalFusionTransformer(nn.Module):
    def __init__(
            self, 
            input_window, 
            output_window, 
            d_model, 
            dropout,
            static_cate_num,
            future_cate_num,
            category_num,
            continous_input_size,
            static_num_input,
            encoder_num_input,
            decoder_num_input,
            num_heads,
            quantile,
            tau
            ):
        
        super(TemporalFusionTransformer, self).__init__()

        self.sequence_len = input_window + output_window
        self.dropout = dropout

        # embedding
        self.static_emb = CategoricalEmbedding(d_model, static_cate_num)
        self.future_emb = CategoricalEmbedding(d_model, future_cate_num)
        self.category_emb = CategoricalEmbedding(d_model, category_num)
        self.continuous_emb = ContinuousEmbedding(d_model, continous_input_size)

        self.static_variable_selection = VariableSelection(d_model, static_num_input)
        self.encoder_variable_selection = VariableSelection(d_model, encoder_num_input)
        self.decoder_variable_selection = VariableSelection(d_model, decoder_num_input)

        # static covariate encoders
        self.grn_cs = GRN(d_model, d_model, d_model, dropout) # variable selection context
        self.grn_cc = GRN(d_model, d_model, d_model, dropout) # lstm initial cell state
        self.grn_ch = GRN(d_model, d_model, d_model, dropout) # lstm initial hidden state
        self.grn_ce = GRN(d_model, d_model, d_model, dropout) # static enrichment context

        # lstm encoder-decoder
        self.lstm_encoder = nn.LSTM(d_model, d_model, num_layers=1, batch_first=True)
        self.lstm_decoder = nn.LSTM(d_model, d_model, num_layers=1, batch_first=True)

        self.locality_glu = GLU(d_model, d_model)
        self.locality_layernorm = nn.LayerNorm(d_model)

        # static enrichment
        self.static_enrichment_grn = GRN(d_model, d_model, d_model, dropout)

        # temporal self-attention
        self.mask = torch.ones(self.sequence_len, self.sequence_len, dtype=torch.bool, device=device).triu(diagonal=1)
        self.imha = InterpretableMultiheadAttention(d_model, num_heads, dropout)
        self.attention_glu = GLU(d_model, d_model)
        self.attention_layernorm = nn.LayerNorm(d_model)

        # position-wise feed-forward
        self.position_wise_grn = GRN(d_model, d_model, d_model, dropout)
        self.position_wise_glu = GLU(d_model, d_model)
        self.position_wise_layernorm = nn.LayerNorm(d_model)

        # quantile output
        self.quantile_output = QuantileOutput(d_model, quantile, tau)

    def forward(self, 
                static_input, 
                future_input, 
                category_input, 
                continuous_input
                ):
        
        # embedding
        # shape : (batch_size, sequence_length, num_of_features, embedding_dim)        
        static_input = self.static_emb(static_input) # (batch,1,static_num,d_model)
        future_input = self.future_emb(future_input) # (batch,decoder_len,future_num,d_model)
        category_input = self.category_emb(category_input) # (batch,encoder_len,cate_input_num,d_model)
        continuous_input = self.continuous_emb(continuous_input) # (batch,encoder_len,conti_input_num,d_model)

        # static covariates encoders
        static_encoder_input = self.static_variable_selection(static_input)
        c_s = self.grn_cs(static_encoder_input)
        c_c = self.grn_cc(static_encoder_input)
        c_h = self.grn_ch(static_encoder_input)
        c_e = self.grn_ce(static_encoder_input)

        # lstm encoder-decoder
        past_input = torch.cat([category_input,continuous_input], dim=2)
        encoder_input = self.encoder_variable_selection(past_input, c_s)
        decoder_input = self.decoder_variable_selection(future_input, c_s)
        resid_vs = torch.cat([encoder_input, decoder_input], dim=1)

        encoder_output, (e_c_c, e_c_h) = self.lstm_encoder(encoder_input, (c_c.permute(1,0,-1), c_h.permute(1,0,-1)))
        decoder_output, _ = self.lstm_decoder(decoder_input, (e_c_c, e_c_h))
        enc_dec_output = torch.cat([encoder_output, decoder_output], dim=1)

        enc_dec_glu_output = self.locality_glu(enc_dec_output)
        static_enrich_input = self.locality_layernorm(resid_vs + enc_dec_glu_output)

        # static enrichment
        query = self.static_enrichment_grn(static_enrich_input, c_e)
        key = self.static_enrichment_grn(static_enrich_input, c_e)
        value = self.static_enrichment_grn(static_enrich_input, c_e)

        # temporal self-attention
        imha_output, attention = self.imha(query, key, value, self.mask)
        # gate, add&norm 할 차례



