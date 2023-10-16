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