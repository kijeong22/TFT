import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random
import os
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import matplotlib.gridspec as gridspec
import wandb
from scipy import stats

from model import TemporalFusionTransformer
from preprocess import *
from train import train
from eval import eval
from valid import valid
from metric import QuantileLoss

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
    parser.add_argument("-b", type=int, default=[20], nargs=1, help="batch_size")
    parser.add_argument("-lr", type=float, default=[1e-2], nargs=1, help="learning_rate")
    parser.add_argument("-es", type=int, default=[20], nargs=1, help="early_stopping")
    parser.add_argument("-el", type=int, default=[24], nargs=1, help="encoder_len")
    parser.add_argument("-dl", type=int, default=[5], nargs=1, help="decoder_len")
    parser.add_argument("-dm", type=int, default=[40], nargs=1, help="d_model")
    parser.add_argument("-dr", type=float, default=[0.1], nargs=1, help="dropout")
    parser.add_argument("-nh", type=int, default=[8], nargs=1, help="num_heads")
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
    key = args.key
    name = args.name

    if key is not None:
        
        wandb.login(key=key)
        wandb.init(project='TFT', name=name)

    seed_everything(seed)

    df = load_data()
    train_set, columns = preprocess(df)
    train_set, valid_set, test_set = split_data(train_set)

    static_variables = list(train_set.columns[:3])
    future_variables = list(train_set.columns[3:7])
    past_categorical_variables = list(train_set.columns[3:7])
    past_continuous_variables = list(train_set.columns[7:])
    target = train_set.columns[-1]

    train_temp_set = TemporalFusionDataset(train_set, encoder_len, decoder_len, static_variables, future_variables,
                                                past_categorical_variables, past_continuous_variables, target)
    valid_temp_set = TemporalFusionDataset(valid_set, encoder_len, decoder_len, static_variables, future_variables,
                                                past_categorical_variables, past_continuous_variables, target)
    test_temp_set = TemporalFusionDataset(test_set, encoder_len, decoder_len, static_variables, future_variables,
                                                past_categorical_variables, past_continuous_variables, target)
    
    train_loader, valid_loader, test_loader = loader(train_temp_set, valid_temp_set, test_temp_set, batch_size=batch_size)

    static_cate_num = [2,2,2]
    future_cate_num = [32,7,24,2] # day, dayofweek, hour, holiday
    category_num = [32,7,24,2] # day, dayofweek, hour, holiday
    continuous_input_size = 7
    static_num_input = 3
    encoder_num_input = 11
    decoder_num_input = 4
    tau = decoder_len
    quantiles = [0.1,0.5,0.9]

    model = TemporalFusionTransformer(encoder_len=encoder_len, 
                                  decoder_len=decoder_len, 
                                  d_model=d_model, 
                                  dropout=dropout,
                                  static_cate_num=static_cate_num, 
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
                torch.save(model.state_dict(), 'best_tft.pth')
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
    
    model.load_state_dict(torch.load('best_tft.pth'))

    # eval_metric = NGLLLoss().to(device)

    # test_loss, test_pred, mu_pred, sigma_pred = eval(model, test_loader, eval_metric, device)

    # print('Test Metric : Negative Gaussian Log Likelihood', f'\nTest loss : {test_loss}')

    # test_pred = test_pred[0].squeeze(0)
    # test_true = next(iter(test_loader))[1].squeeze(0)

    # ci = stats.norm.interval(0.95, loc=mu_pred.squeeze(0).cpu(), scale=sigma_pred.squeeze(0).cpu())

    # grid = gridspec.GridSpec(input_size,1)
    # plt.figure(figsize=(10,25))
    # plt.subplots_adjust(hspace=0.3)

    # for idx, column in enumerate(df.columns):
    #     ax = plt.subplot(grid[idx])
        
    #     ax.plot(test_pred[:,idx].cpu(), label='pred')
    #     ax.plot(test_true[:,idx].cpu(), label='true')

    #     ax.fill_between(range(0,30), y1=ci[0][:,idx], y2=ci[1][:,idx], facecolor='skyblue', alpha = 0.2)

    #     ax.legend()
    #     plt.title(f'{column}', fontsize=13)
    #     plt.xlabel('Date')
    #     plt.ylabel('Value')

    # plt.savefig('final_predict_plot.png')
    # plt.show()
    if key is not None:

        #wandb.log({"True vs Pred": wandb.Image("final_predict_plot.png")})
        wandb.finish()

if __name__ == "__main__":
    main()