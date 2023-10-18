import torch

def eval(model, data_loader, criterion, device):

    model.eval()

    predictions = []
    total_loss = []
    targets = []

    with torch.no_grad():
        for idx, (static_cate, future, past_category, past_continuous, target) in enumerate(data_loader):

            static_cate = static_cate.to(device)
            #static_conti = static_conti.to(device)
            future = future.to(device)
            past_category = past_category.to(device)
            past_continuous = past_continuous.to(device)
            target = target.to(device)

            pred, _ = model(static_cate, future, past_category, past_continuous)

            loss = criterion(target, pred)

            total_loss.append(loss)

            condition = static_cate.item()

            if condition >= 25:

                condition = condition % 24

                if condition == 0:
                    
                    condition = 24
            
            if idx % len(future[0,:,0]) == condition-1: # len(future[0,:,0]) == decoder_len

                predictions.append(pred)
                targets.append(target)

    return sum(total_loss)/len(total_loss), predictions, targets

