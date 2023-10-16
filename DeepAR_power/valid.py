import torch

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

            mu_seq, sigma_seq, _ = model(cate, enc_conti, dec_conti, future, train_mode=False)

            loss = criterion(mu_seq[:,:,-1], sigma_seq[:,:,-1], target)

            total_loss.append(loss)

    return sum(total_loss) / len(total_loss)