import torch

def eval(model, data_loader, criterion, device):

    model.eval()

    predictions = []
    mus = []
    sigmas = []
    total_loss = []
    targets = []

    with torch.no_grad():
        for idx, (cate, enc_conti, dec_conti, future, target) in enumerate(data_loader):

            cate = cate.to(device) 
            enc_conti = enc_conti.to(device)
            dec_conti = dec_conti.to(device)
            future = future.to(device)
            target = target.to(device)

            mu_seq, sigma_seq, output_seq = model(cate, enc_conti, dec_conti, future, train_mode=False)

            mu = mu_seq[:,:,-1] # (batch, decoder_len)
            sigma = sigma_seq[:,:,-1] # (1, decoder_len)
            pred = output_seq[:,:,-1] # (1, decoder_len)

            loss = criterion(mu, sigma, target)

            total_loss.append(loss)

            condition = cate[0,0,0].item()

            if condition >= 25:

                condition = condition % 24

                if condition == 0:
                    
                    condition = 24
            
            if idx % len(future[0,:,0]) == condition-1: # len(future[0,:,0]) == decoder_len

                predictions.append(pred)
                targets.append(target)

    return sum(total_loss)/len(total_loss), predictions, mus, sigmas, targets

