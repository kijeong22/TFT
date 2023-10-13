import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import datetime

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
        cdh_temp = cdh_df['temperature'].values
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


def split_data(data):

    train_list = []
    valid_list = []
    test_list = []

    for i in range(len(data['building_num'].unique())):

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


class TemporalFusionPowerDataset(Dataset):
    def __init__(self,
                 data,
                 encoder_len:int,
                 decoder_len:int,
                 static_categorical_variables:list[str],
                 static_continuous_variables:list[str],
                 future_variables:list[str],
                 past_categorical_variables:list[str],
                 past_continuous_variables:list[str],
                 target:str,
                 stride=1
                 ):
        super(TemporalFusionPowerDataset, self).__init__()
        
        self.data = data
        self.encoder_len = encoder_len
        self.decoder_len = decoder_len
        self.stride = stride
        self.sequence_len = self.encoder_len + self.decoder_len # 192

        self.num_buildings = len(data[static_categorical_variables].iloc[:,0].unique()) # 100
        self.each_build_seq_data_len = len(self.data) // self.num_buildings - self.sequence_len + 1
        # train : 1503
        # valid : 145
        # test : 145

        self.static_cate_data = data[static_categorical_variables]
        self.static_conti_data = data[static_continuous_variables]
        self.future_data = data[future_variables]
        self.past_cate_data = data[past_categorical_variables]
        self.past_conti_data = data[past_continuous_variables]
        self.target = data[target]

    def __len__(self):

        data_len = ((len(self.data) // self.num_buildings - self.sequence_len) // self.stride + 1) * self.num_buildings

        return data_len

    def __getitem__(self, idx):

        new_idx = idx + (idx//self.each_build_seq_data_len) * (self.sequence_len-1)
                
        static_cate_data = torch.tensor(self.static_cate_data[new_idx:new_idx+1].to_numpy())
        static_conti_data = torch.tensor(self.static_conti_data[new_idx:new_idx+1].to_numpy())
        future_data = torch.tensor(self.future_data[new_idx+self.encoder_len:new_idx+self.sequence_len].to_numpy())
        past_cate_data = torch.tensor(self.past_cate_data[new_idx:new_idx+self.encoder_len].to_numpy())
        past_conti_data = torch.tensor(self.past_conti_data[new_idx:new_idx+self.encoder_len].to_numpy())
        target = torch.tensor(self.target[new_idx+self.encoder_len:new_idx+self.sequence_len].to_numpy())
        
        return static_cate_data, static_conti_data, future_data, past_cate_data, past_conti_data, target
