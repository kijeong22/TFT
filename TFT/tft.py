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

# preprocessing
def fdr_data(code, start, end):
    
    df = fdr.DataReader(code, start, end)
    
    return df

def preprocessing(data, input_window, output_window):

    data['date'] = data.index
    data['day'] = data.date.dt.day
    data['month'] = df.date.dt.month
    data['dayofweek'] = df.date.dt.day_of_week
    data.drop(columns='date', inplace=True)

    train_set, test_set = train_test_split(data, test_size=output_window, shuffle=False)
    train_set, valid_set = train_test_split(train_set, test_size=output_window, shuffle=False)

    valid_set = pd.concat([train_set[-input_window:], valid_set])
    test_set = pd.concat([valid_set[-input_window:], test_set])

    return train_set, valid_set, test_set

class WindowDataset(Dataset):
    def __init__(self, data, input_window, output_window, input_size, stride=1):
        
        L = data.shape[0]
        num_samples = (L - input_window - output_window) // stride + 1
        data_tensor = torch.FloatTensor(data.to_numpy())

        X = torch.zeros(num_samples, input_window, input_size)
        y = torch.zeros(num_samples, output_window, input_size)

        for i in range(num_samples):
            
            X[i,:] = data_tensor[i*stride : i*stride+input_window] # (encoder_len, num_feature)
            y[i,:] = data_tensor[i*stride+input_window : i*stride+input_window+output_window] # (decoder_len, num_feature)

        self.x = X
        self.y = y
        self.len = len(X)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.len
    
code = '005930'
start = '2010'
end = '2023'
input_window = 60
output_window = 5
batch_size = 20

df = fdr_data(code, start, end)    
train_set, valid_set, test_set = preprocessing(df, input_window, output_window)

input_size = train_set.shape[1]
batch_size = batch_size
output_size = input_size

train_set = WindowDataset(train_set, input_window, output_window, input_size)
valid_set = WindowDataset(valid_set, input_window, output_window, input_size)
test_set = WindowDataset(test_set, input_window, output_window, input_size)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

train_input = next(iter(train_loader))[0]
train_target = next(iter(train_loader))[1]

train_input.shape # (20,60,9)
train_target.shape # (20,5,9)

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

d_model = 40 # hidden
category_num = [32,13,7]

category_input = torch.tensor(train_input[:,:,-3:], dtype=int)
continuous_input = train_input[:,:,:6]

cat_emb = CategoricalEmbedding(d_model, category_num)
con_emb = ContinuousEmbedding(d_model, continuous_input.shape[-1])

category_input = cat_emb(category_input)
continuous_input = con_emb(continuous_input)

category_input.shape
continuous_input.shape

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

        if self.input_size != self.d_model:
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

ksi = category_input
varselec = VariableSelection(d_model, ksi.shape[-2])
out = varselec(ksi)
out.shape


# class StaticCovariateEncoder(nn.Moduel):
#     def __init__(self, ):
#         super(StaticCovariateEncoder, self).__init__()

#     def forward(self, ):


#         return c_s, c_c, c_h, c_e


class InterpretableMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_attn = self.embed_dim / self.num_heads
        self.d_v = self.d_attn

        self.softmax = nn.Softmax()


    def forward(query, key, value)
