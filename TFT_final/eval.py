import torch

def eval(model, data_loader, criterion, device):

    model.eval()

    predictions = []
    total_loss = []
    targets = []

    with torch.no_grad():
        for static, future, past_category, past_continuous, target in data_loader:

            static = static.to(device)
            future = future.to(device)
            past_category = past_category.to(device)
            past_continuous = past_continuous.to(device)
            target = target.to(device)

            pred, _ = model(static, future, past_category, past_continuous)

            loss = criterion(target, pred[:,:,1])

            total_loss.append(loss)
            predictions.append(pred)
            targets.append(target)

    return sum(total_loss)/len(total_loss), predictions, targets

