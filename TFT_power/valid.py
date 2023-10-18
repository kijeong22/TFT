import torch

def valid(model, data_loader, criterion, device):

    model.eval()

    total_loss = []

    with torch.no_grad():
        for static_cate, future, past_category, past_continuous, target in data_loader:

            static_cate = static_cate.to(device)
            #static_conti = static_conti.to(device)
            future = future.to(device)
            past_category = past_category.to(device)
            past_continuous = past_continuous.to(device)
            target = target.to(device)

            pred, _ = model(static_cate, future, past_category, past_continuous)

            loss = criterion(target, pred)

            total_loss.append(loss)

    return sum(total_loss) / len(total_loss)