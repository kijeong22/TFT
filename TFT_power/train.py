def train(model, data_loader, optimizer, criterion, device, batch_size):

    model.train()

    total_loss = []

    for static_cate, future, past_category, past_continuous, target in data_loader:

        static_cate = static_cate.to(device)
        #static_conti = static_conti.to(device)
        future = future.to(device)
        past_category = past_category.to(device)
        past_continuous = past_continuous.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        pred, _ = model(static_cate, future, past_category, past_continuous)

        loss = criterion(target, pred)

        loss.backward()
        optimizer.step()

        if target.shape[0] == batch_size:
            weighted_loss = loss
        else:
            weighted_loss = loss * batch_size / target.shape[0]

        total_loss.append(weighted_loss)
    
    return sum(total_loss) / len(total_loss)