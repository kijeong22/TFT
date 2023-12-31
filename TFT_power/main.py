import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import wandb


from model import TemporalFusionTransformer
from preprocess import *
from train import train
from eval import eval
from valid import valid
from metric import QuantileLoss, smape

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def argparse_custom():
    parser = argparse.ArgumentParser(description="TFT")

    parser.add_argument("-s", type=int, default=[42], nargs=1, help="seed")
    parser.add_argument("-e", type=int, default=[100], nargs=1, help="epoch")
    parser.add_argument("-b", type=int, default=[64], nargs=1, help="batch_size")
    parser.add_argument("-lr", type=float, default=[1e-2], nargs=1, help="learning_rate")
    parser.add_argument("-es", type=int, default=[15], nargs=1, help="early_stopping")
    parser.add_argument("-el", type=int, default=[168], nargs=1, help="encoder_len")
    parser.add_argument("-dl", type=int, default=[24], nargs=1, help="decoder_len")
    parser.add_argument("-dm", type=int, default=[8], nargs=1, help="d_model")
    parser.add_argument("-dr", type=float, default=[0.1], nargs=1, help="dropout")
    parser.add_argument("-nh", type=int, default=[2], nargs=1, help="num_heads")
    parser.add_argument("-bmn", type=str, default=['best_tft_power'], nargs=1, help="best model name")
    parser.add_argument("-key", type=str, default=[None], nargs=1, help="wandb login key")
    parser.add_argument("-name", type=str, default=['default'], nargs=1, help="wandb project run name")
    
    args = parser.parse_args()

    args.s = args.s[0]
    args.e = args.e[0]
    args.b = args.b[0]
    args.lr = args.lr[0]
    args.es = args.es[0]
    args.el = args.el[0]
    args.dl = args.dl[0]
    args.dm = args.dm[0]
    args.dr = args.dr[0]
    args.nh = args.nh[0]
    args.bmn = args.bmn[0]
    args.key = args.key[0]
    args.name = args.name[0]

    return args

def main():

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    args = argparse_custom()
    seed = args.s
    epochs = args.e
    batch_size = args.b
    learning_rate = args.lr
    early_stopping = args.es
    encoder_len = args.el
    decoder_len = args.dl
    d_model = args.dm
    dropout = args.dr
    num_heads = args.nh
    best_model_name = args.bmn
    key = args.key
    name = args.name

    if key is not None:
        
        wandb.login(key=key)
        wandb.init(project='TFT_power_each_scaling', name=name)

    seed_everything(seed)

    df = load_data()
    train_set, columns = preprocess(df)
    train_set, valid_set, test_set = split_data(train_set)

    train_set, valid_set, test_set, scaler_list = scaler(train_set, valid_set, test_set)

    static_categorical_variables = list(train_set.columns[:1])
    #static_continuous_variables = list(train_set.columns[1:3])
    future_variables = list(train_set.columns[1:5])
    past_categorical_variables = list(train_set.columns[1:5])
    past_continuous_variables = list(train_set.columns[5:])
    target = train_set.columns[-1]

    train_temp_set = TemporalFusionPowerDataset(train_set, encoder_len, decoder_len, static_categorical_variables, 
                                                future_variables, past_categorical_variables, past_continuous_variables, target)
    valid_temp_set = TemporalFusionPowerDataset(valid_set, encoder_len, decoder_len, static_categorical_variables, 
                                                future_variables, past_categorical_variables, past_continuous_variables, target)
    test_temp_set = TemporalFusionPowerDataset(valid_set, encoder_len, decoder_len, static_categorical_variables,
                                                future_variables, past_categorical_variables, past_continuous_variables, target)

    train_loader, valid_loader, test_loader = loader(train_temp_set, valid_temp_set, test_temp_set, batch_size=batch_size)

    static_cate_num = [101]
    #static_conti_size = 2
    future_cate_num = [32,7,24,2] # day, dayofweek, hour, holiday
    category_num = [32,7,24,2] # day, dayofweek, hour, holiday
    continuous_input_size = 7
    static_num_input = 1
    encoder_num_input = 11
    decoder_num_input = 4
    tau = decoder_len
    quantiles = [0.1,0.5,0.9]

    model = TemporalFusionTransformer(encoder_len=encoder_len,
                                  decoder_len=decoder_len,
                                  d_model=d_model, 
                                  dropout=dropout,
                                  static_cate_num=static_cate_num,
                                  #static_conti_size=static_conti_size,
                                  future_cate_num=future_cate_num, 
                                  category_num=category_num,
                                  continuous_input_size=continuous_input_size, 
                                  static_num_input=static_num_input, 
                                  encoder_num_input=encoder_num_input,
                                  decoder_num_input=decoder_num_input, 
                                  num_heads=num_heads, 
                                  quantiles=quantiles, 
                                  tau=tau,
                                  device=device).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = QuantileLoss(quantiles=quantiles).to(device)

    best_valid_loss = float('inf')
    early_stopping_count = 0

    with tqdm(range(1, epochs+1)) as tr:
        for epoch in tr:

            train_loss = train(model, train_loader, optimizer, criterion, device, batch_size)
            valid_loss = valid(model, valid_loader, criterion, device)

            if epoch % 10 == 0:
                print(f'epoch:{epoch}, train_loss:{train_loss.item():5f}')
                print(f'epoch:{epoch}, valid_loss:{valid_loss.item():5f}')

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), f'best_model/{best_model_name}.pth')
                early_stopping_count = 0
            else:
                early_stopping_count += 1
            
            if early_stopping_count >= early_stopping:
                print(f'epoch:{epoch}, train_loss:{train_loss.item():5f}')
                print(f'epoch:{epoch}, valid_loss:{valid_loss.item():5f}')
                print(f'best valid loss :{best_valid_loss}')
                break

            if key is not None:
                wandb.log({
                    "Train Loss": train_loss.item(),
                    "Valid Loss": valid_loss.item()})

    model = TemporalFusionTransformer(encoder_len=encoder_len,
                                  decoder_len=decoder_len,
                                  d_model=d_model, 
                                  dropout=dropout,
                                  static_cate_num=static_cate_num,
                                  #static_conti_size=static_conti_size,
                                  future_cate_num=future_cate_num, 
                                  category_num=category_num,
                                  continuous_input_size=continuous_input_size, 
                                  static_num_input=static_num_input, 
                                  encoder_num_input=encoder_num_input,
                                  decoder_num_input=decoder_num_input, 
                                  num_heads=num_heads, 
                                  quantiles=quantiles, 
                                  tau=tau,
                                  device=device).to(device)
    
    model.load_state_dict(torch.load(f'best_model/{best_model_name}.pth'))

    test_loss, test_pred, test_target = eval(model, test_loader, criterion, device)
    
    pred = torch.cat(test_pred, dim=1).squeeze().unsqueeze(-1)
    target = torch.cat(test_target, dim=1).squeeze().unsqueeze(-1)

    smape_score = smape(target, pred[:,1])

    result = []
    for i in tqdm(range(100)):

        pred_10 = pred[168*i:168*i+168,0].cpu().detach().numpy() * scaler_list[i]
        pred_50 = pred[168*i:168*i+168,1].cpu().detach().numpy() * scaler_list[i]
        pred_90 = pred[168*i:168*i+168,2].cpu().detach().numpy() * scaler_list[i]
        true_ = target[168*i:168*i+168].cpu().detach().numpy() * scaler_list[i]

        result.append([pred_10, pred_50, pred_90, true_])

    x_values = pd.to_datetime(df['일시'][-168:], format='%Y%m%d %H')
    for i in range(100):

        pred_10 = result[i][0]
        pred_50 = result[i][1]
        pred_90 = result[i][2]
        true = result[i][3]

        plt.figure(figsize=(15, 7))
        plt.plot(x_values, pred_50, label='Pred', color='blue', marker='o', markersize=3)
        
        for t in range(len(x_values)):
            plt.vlines(x=x_values.iloc[t], ymin=pred_10[t], ymax=pred_90[t], color='blue', alpha=0.2)
        
        plt.plot(x_values, true, label='True', color='red', marker='o', markersize=3)

        plt.legend(fontsize=15)
        plt.xlabel('Date', fontsize=13)
        plt.ylabel('Value', fontsize=13)
        plt.show()

        if key is not None:

            wandb.log({f"building {i+1}": wandb.Image(plt)})
    
    if key is not None:

        wandb.log({'Test Loss' : test_loss.item(), 'SMAPE' : smape_score.item()})
        wandb.finish()


if __name__ == "__main__":
    main()