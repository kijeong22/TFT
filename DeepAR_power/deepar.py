import torch
import torch.nn as nn
import torch.optim as optim
import FinanceDataReader as fdr
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import datetime
from tqdm import tqdm
import random
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.gridspec as gridspec
import wandb

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

def load_data():

    df = pd.read_csv('train.csv')
    building = pd.read_csv('building_info.csv')
    df = pd.merge(df, building, on='건물번호')

    return df


def fill_missing_with_avg(df, columns):
    for i in range(len(df)):
        if pd.isna(df.loc[i, columns]):
            
            prev_value_sum = df.loc[i-4:i-1, columns].sum()
            next_value_sum = df.loc[i+1:i+4, columns].sum()
            avg_value = (prev_value_sum + next_value_sum) / 8

            df.loc[i, columns] = avg_value
    return df

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

    train_set['CDH'] = CDH(train_set, 100)

    train_set = train_set.drop(columns=['date'])
    train_set = pd.concat([train_set.iloc[:,0:5], train_set.iloc[:,6:], train_set.iloc[:,5:6]], axis=1)

    train_set = train_set[[
        'building_num', 'total_area', 'cooling_area', # static variable
        'day', 'weekday', 'hour', 'holiday', # future variable
        'temperature', 'precipitation', 'windspeed', 'humidity', 'DI', 'CDH', # observed variable
        'power_consumption' # target
        ]]

    return train_set, train_set.columns


class DeepARPowerDataset(Dataset):
    def __init__(self,
                 data,
                 encoder_len:int,
                 decoder_len:int,
                 categorical_variables:list[str],
                 continuous_variables:list[str],
                 future_variables:list[str],
                 target:str,
                 stride=1
                 ):
        super(DeepARPowerDataset, self).__init__()
        
        self.data = data
        self.encoder_len = encoder_len
        self.decoder_len = decoder_len
        self.stride = stride
        self.sequence_len = self.encoder_len + self.decoder_len # 192

        self.num_buildings = len(data[categorical_variables].iloc[:,0].unique()) # 100
        self.each_build_seq_data_len = len(self.data) // self.num_buildings - self.sequence_len + 1
        # train : 1503
        # valid : 145
        # test : 145

        self.cate_data = data[categorical_variables]
        self.conti_data = data[continuous_variables]
        self.future_data = data[future_variables]
        self.target = data[target]

    def __len__(self):

        data_len = ((len(self.data) // self.num_buildings - self.sequence_len) // self.stride + 1) * self.num_buildings

        return data_len

    def __getitem__(self, idx):

        new_idx = idx + (idx//self.each_build_seq_data_len) * (self.sequence_len-1)
                
        cate_data = torch.tensor(self.cate_data[new_idx:new_idx+self.encoder_len].to_numpy())
        enc_conti_data = torch.tensor(self.conti_data[new_idx:new_idx+self.encoder_len].to_numpy())
        dec_conti_data = torch.tensor(self.conti_data[new_idx+self.encoder_len:new_idx+self.sequence_len].to_numpy())
        future_data = torch.tensor(self.future_data[new_idx+self.encoder_len:new_idx+self.sequence_len].to_numpy())
        target = torch.tensor(self.target[new_idx+self.encoder_len:new_idx+self.sequence_len].to_numpy())
        
        return cate_data, enc_conti_data, dec_conti_data, future_data, target


def split_data(data):

    train_list = []
    valid_list = []
    test_list = []

    for i in tqdm(range(len(data['building_num'].unique()))):

        train, test = train_test_split(data[data['building_num'] == i+1], test_size=24*7, shuffle=False)
        train, valid = train_test_split(train[train['building_num'] == i+1], test_size=24*7, shuffle=False)
        valid = pd.concat([train[-24*7:], valid]).reset_index(drop=True)
        test = pd.concat([valid[-24*7:], test]).reset_index(drop=True)

        train_list.append(train)
        valid_list.append(valid)
        test_list.append(test)

    train_set = pd.concat(train_list).reset_index(drop=True)
    valid_set = pd.concat(valid_list).reset_index(drop=True)
    test_set = pd.concat(test_list).reset_index(drop=True)

    return train_set, valid_set, test_set

def loader(train_set, valid_set, test_set, batch_size):

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    return train_loader, valid_loader, test_loader

df = load_data()

data, columns = preprocess(df)

train_set, valid_set, test_set = split_data(data)


def scaler(train_set, valid_set, test_set):

    scaler_list = []
    train_set_s = pd.DataFrame()
    valid_set_s = pd.DataFrame()
    test_set_s = pd.DataFrame()
    scaling_features = ['total_area', 'cooling_area', 'temperature', 'precipitation', 'windspeed', 'humidity', 'DI', 'CDH']
    scaling_target = ['power_consumption']
    for i in range(1, len(train_set['building_num'].unique())+1):

        scaler_feature = MinMaxScaler()
        scaler_target = MinMaxScaler()

        train_set_each_building = train_set[train_set['building_num']==i]
        valid_set_each_building = valid_set[valid_set['building_num']==i]
        test_set_each_building = test_set[test_set['building_num']==i]

        train_set_each_building[scaling_features] = scaler_feature.fit_transform(train_set_each_building[scaling_features])
        valid_set_each_building[scaling_features] = scaler_feature.transform(valid_set_each_building[scaling_features])
        test_set_each_building[scaling_features] = scaler_feature.transform(test_set_each_building[scaling_features])

        train_set_each_building[scaling_target] = scaler_target.fit_transform(train_set_each_building[scaling_target])
        valid_set_each_building[scaling_target] = scaler_target.transform(valid_set_each_building[scaling_target])
        test_set_each_building[scaling_target] = scaler_target.transform(test_set_each_building[scaling_target])

        train_set_s = pd.concat([train_set_s, train_set_each_building])
        valid_set_s = pd.concat([valid_set_s, valid_set_each_building])
        test_set_s = pd.concat([test_set_s, test_set_each_building])

        scaler_list.append(scaler_target)

    return train_set_s, valid_set_s, test_set_s, scaler_list

train_set, valid_set, test_set, scaler_list = scaler(train_set, valid_set, test_set)

encoder_len = 168 # 7 day
decoder_len = 24 # 1 day
batch_size = 64


categorical_variables = list(train_set.columns[0:1]) + list(train_set.columns[3:7])
continuous_variables = list(train_set.columns[1:3]) + list(train_set.columns[7:])
future_variables = list(train_set.columns[3:7])
target = train_set.columns[-1]


train_temp_set = DeepARPowerDataset(train_set, encoder_len, decoder_len, categorical_variables, 
                                    continuous_variables, future_variables, target)
valid_temp_set = DeepARPowerDataset(valid_set, encoder_len, decoder_len, categorical_variables, 
                                    continuous_variables, future_variables, target)
test_temp_set = DeepARPowerDataset(valid_set, encoder_len, decoder_len, categorical_variables, 
                                    continuous_variables, future_variables, target)

train_loader, valid_loader, test_loader = loader(train_temp_set, valid_temp_set, test_temp_set, batch_size=batch_size)


train_cate = next(iter(train_loader))[0]
train_enc_conti = next(iter(train_loader))[1]
train_dec_conti = next(iter(train_loader))[2]
train_future = next(iter(train_loader))[3]
train_target = next(iter(train_loader))[4]


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


class Encoder(nn.Module):
    def __init__(self, conti_size, d_model, embedding_dim, category_num:list, num_layers=2, dropout=0.1):
        super(Encoder, self).__init__()

        self.embedding = CategoricalEmbedding(embedding_dim, category_num)
        self.lstm = nn.LSTM(conti_size + embedding_dim*len(category_num), d_model, num_layers, dropout=dropout, batch_first=True)

    def forward(self, cate, conti):

        cate_embed = self.embedding(cate.to(torch.int))
        cate_embed = cate_embed.view(conti.shape[0], conti.shape[1], -1) # (batch, encoder_len, num_cate_features*embedding_dim)

        encoder_input = torch.cat([cate_embed, conti], dim=-1)

        _, (hidden, cell) = self.lstm(encoder_input.to(torch.float))

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, conti_size, d_model, embedding_dim, category_num:list, device,
                 num_layers=2, dropout=0.1, t_forcing=1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.output_size = conti_size
        self.t_forcing = t_forcing
        self.device = device

        self.embedding = CategoricalEmbedding(embedding_dim, category_num)
        self.lstm = nn.LSTM(conti_size + embedding_dim*len(category_num), self.d_model, self.num_layers, dropout=dropout, batch_first=True)

        self.mu_linear1 = nn.Linear(self.num_layers, 1)
        self.mu_linear2 = nn.Linear(self.d_model, self.output_size)

        self.sigma_linear1 = nn.Linear(self.num_layers, 1)
        self.sigma_linear2 = nn.Linear(self.d_model, self.output_size)
        self.softplus = nn.Softplus()

    def forward(self, past_conti, dec_conti, future, hidden, cell, train_mode):

        decoder_len = dec_conti.shape[1]
        batch = dec_conti.shape[0]

        mu_seq = torch.zeros(batch, decoder_len, self.output_size, device=self.device)
        sigma_seq = torch.zeros(batch, decoder_len, self.output_size, device=self.device)
        output_seq = torch.zeros(batch, decoder_len, self.output_size, device=self.device)

        future_embed = self.embedding(future)
        future_embed = future_embed.view(batch, decoder_len, -1) # (batch, decoder_len, num_future_features*embedding_dim)

        past_conti = past_conti[:,-2:-1,:]

        for t in range(decoder_len):

            decoder_input = torch.cat([past_conti, future_embed[:,t:t+1,:]], dim=-1).to(torch.float)
            
            _, (hidden, cell) = self.lstm(decoder_input, (hidden, cell))

            mu = self.mu_linear1(hidden.view(batch, self.d_model, self.num_layers)) # (batch, d_model, 1)
            mu = self.mu_linear2(mu.squeeze(-1)) # (batch, conti_size)

            sigma = self.sigma_linear1(hidden.view(batch, self.d_model, self.num_layers)) # (batch, d_model, 1)
            sigma = self.sigma_linear2(sigma.squeeze(-1)) # (batch, conti_size)
            sigma = self.softplus(sigma)

            mu_seq[:,t,:] = mu
            sigma_seq[:,t,:] = sigma

            likelihood = torch.distributions.normal.Normal(mu, sigma)
            output = likelihood.sample()

            output_seq[:,t,:] = output
            past_conti = output.unsqueeze(1)

            if train_mode:

                if random.random() < self.t_forcing:
                    past_conti = dec_conti[:, t:t+1, :]

        return mu_seq, sigma_seq, output_seq
    
class DeepAR(nn.Module):
    def __init__(self, conti_size, d_model, embedding_dim, enc_category_num, dec_category_num,
                 device, num_layers, dropout, t_forcing):
        super(DeepAR, self).__init__()

        self.encoder = Encoder(conti_size, d_model, embedding_dim, enc_category_num, num_layers, dropout)
        self.decoder = Decoder(conti_size, d_model, embedding_dim, dec_category_num, device, 
                               num_layers, dropout, t_forcing)

    def forward(self, cate, enc_conti, dec_conti, future, train_mode=True):

        hidden, cell = self.encoder(cate, enc_conti)
        mu_seq, sigma_seq, output_seq = self.decoder(enc_conti, dec_conti, future, hidden, cell, train_mode)

        return mu_seq, sigma_seq, output_seq


conti_size = 9
d_model = 32
embedding_dim = 32
enc_category_num = [101,32,7,24,2]
dec_category_num = [32,7,24,2]
num_layers = 2
dropout = 0.1
t_forcing = 1

model = DeepAR(conti_size, d_model, embedding_dim, enc_category_num, dec_category_num,
               device, num_layers, dropout, t_forcing).to(device)

mu, sigma, output = model(train_cate.to(device), train_enc_conti.to(device), 
                          train_dec_conti.to(device), train_future.to(device))


class NGLLLoss(nn.Module):
    def __init__(self,):
        super(NGLLLoss, self).__init__()

    def forward(self, mu, sigma, target):

        likelihood = (1/2)*((target-mu)/sigma)**2 + (1/2)*torch.log(torch.tensor(2*torch.pi)) + torch.log(sigma)

        # another way
        # dist = torch.distributions.normal.Normal(mu_seq, sigma_seq)
        # -torch.sum(dist.log_prob(target))

        return torch.sum(likelihood)
    

def train(model, data_loader, optimizer, criterion, device, batch_size):

    model.train()

    total_loss = []

    for cate, enc_conti, dec_conti, future, target in data_loader:

        cate = cate.to(device) 
        enc_conti = enc_conti.to(device)
        dec_conti = dec_conti.to(device)
        future = future.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        mu_seq, sigma_seq, _ = model(cate, enc_conti, dec_conti, future)

        loss = criterion(mu_seq[:,:,-1], sigma_seq[:,:,-1], target)

        loss.backward()
        optimizer.step()

        if target.shape[0] == batch_size:
            weighted_loss = loss
        else:
            weighted_loss = loss * batch_size / target.shape[0]

        total_loss.append(weighted_loss)
    
    return sum(total_loss) / len(total_loss)


def valid(model, data_loader, criterion, device):

    model.eval()

    total_loss = []

    with torch.no_grad():
        for cate, enc_conti, dec_conti, future, target in data_loader:

            cate = cate.to(device) 
            enc_conti = enc_conti.to(device)
            dec_conti = dec_conti.to(device)
            future = future.to(device)
            target = target.to(device)

            mu_seq, sigma_seq, _ = model(cate, enc_conti, dec_conti, future)

            loss = criterion(mu_seq[:,:,-1], sigma_seq[:,:,-1], target)

            total_loss.append(loss)

    return sum(total_loss) / len(total_loss)


learning_rate = 0.01
epochs = 50

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = NGLLLoss().to(device)
best_valid_loss = float('inf')
early_stopping_count = 0
early_stopping = 5


# train_input.shape
# train_target.shape
# train_input[0].shape
# torch.mean(train_input[0], dim=0).shape
# v = (torch.mean(train_input[0], dim=0) + 1).unsqueeze(0)
# v.shape

# mu_seq, sigma_seq, _ = model(train_input.to(device), train_target.to(device), device)

# mu_seq * v.to(device)

# torch.sqrt(v.to(device))

# loss = criterion(mu_seq*v.to(device), sigma_seq*torch.sqrt(v.to(device)), train_target.to(device))
# loss


with tqdm(range(1, epochs+1)) as tr:
    for epoch in tr:

        train_loss = train(model, train_loader, optimizer, criterion, device, batch_size)
        valid_loss = valid(model, valid_loader, criterion, device)

        if epoch % 10 == 0:
            print(f'epoch:{epoch}, train_loss:{train_loss.item():5f}')
            print(f'epoch:{epoch}, valid_loss:{valid_loss.item():5f}')

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best_deepar.pth')
            early_stopping_count = 0
        else:
            early_stopping_count += 1
        
        if early_stopping_count >= early_stopping:
            print(f'epoch:{epoch}, train_loss:{train_loss.item():5f}')
            print(f'epoch:{epoch}, valid_loss:{valid_loss.item():5f}')
            print(f'best valid loss :{best_valid_loss}')
            break


model = DeepAR(conti_size, d_model, embedding_dim, enc_category_num, dec_category_num,
               device, num_layers, dropout, t_forcing).to(device)

model.load_state_dict(torch.load('best_deepar.pth'))

def eval(model, data_loader, criterion, device):

    model.eval()

    predictions = []
    total_loss = []

    with torch.no_grad():
        for input, target, v in data_loader:

            input = input.to(device)
            target = target.to(device)
            v = v.to(device)

            mu_seq, sigma_seq, output_seq = model(input, target, device, train_mode=False)

            mu_scaled = mu_seq * v
            sigma_scaled = sigma_seq * torch.sqrt(v)

            loss = criterion(output_seq, target)

            total_loss.append(loss)
            predictions.append(output_seq)

    return sum(total_loss)/len(total_loss), predictions, mu_scaled, sigma_scaled

# 스케일링 실수해서 급한대로 이렇게라도..

test_loss, test_pred, m, s = eval(model, test_loader, criterion, device)

(test_pred[0]*s+m).squeeze(0)

print('Test Metric : MSE', f'\nTest loss : {test_loss}')

test_pred = test_pred[0].squeeze(0)
test_true = next(iter(test_loader))[1].squeeze(0)


grid = gridspec.GridSpec(6,1)
plt.figure(figsize=(10,25))
plt.subplots_adjust(hspace=0.3)

for idx, column in enumerate(df.columns):
    ax = plt.subplot(grid[idx])
    
    ax.plot((test_pred[0]*s+m).squeeze(0)[:,idx].cpu(), label='pred')
    ax.plot(test_true[:,idx].cpu(), label='true')

    ax.legend()
    plt.title(f'Forecast {column} with DeepAR', fontsize=13)
    plt.xlabel('Date')
    plt.ylabel('Value')