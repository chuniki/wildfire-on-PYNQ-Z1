from tqdm import tqdm
import modules.metrics as metrics
import torch

def eval_fn(loader, model, loss_fn, device):
    
    model.eval()
    loop = tqdm(loader, desc='Validating', leave=True)
    val_losses = []
    smoke_losses = []
    fire_losses = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(device), y.to(device)
        out = model(x)
        val_loss = loss_fn(ground_truth=y, 
                           predictions=out)      
        
        # BCE Loss
        last_losses = loss_fn.get_last_losses()
        val_losses.append(val_loss.item())
        smoke_losses.append(last_losses['smoke_loss'])
        fire_losses.append(last_losses['fire_loss'])
    
        yhat = torch.sigmoid(out.detach())
        metrics.precision_metric.update(yhat, y)
        metrics.recall_metric.update(yhat, y)
        metrics.accuracy_metric.update(yhat, y)
        metrics.f1_metric.update(yhat, y)

    val_mean_loss = sum(val_losses)/len(val_losses)
    smoke_mean_loss = sum(smoke_losses)/len(smoke_losses)
    fire_mean_loss = sum(fire_losses)/len(fire_losses)
    
    precision = metrics.precision_metric.compute()
    recall = metrics.recall_metric.compute()
    accuracy = metrics.accuracy_metric.compute()
    f1 = metrics.f1_metric.compute()
    
    metrics.precision_metric.reset()
    metrics.recall_metric.reset()
    metrics.accuracy_metric.reset()
    metrics.f1_metric.reset()

    print("Total Loss".ljust(12) + "|" + 
          "Smoke Loss".ljust(12) + "|" + 
          "Fire Loss".ljust(12))
    print("------------".ljust(12) + " " + 
          "------------".ljust(12) + " " + 
          "------------".ljust(12))
    print(f'{val_mean_loss:.3f}'.ljust(12) + "|" +
          f'{smoke_mean_loss:.3f}'.ljust(12) + "|" +
          f'{fire_mean_loss:.3f}'.ljust(12))

    print(f'SMOKE -> Precision: {precision[0]:.4f} - Recall: {recall[0]:.4f} - Accuracy: {accuracy[0]:.4f} - F1: {f1[0]:.4f}')
    print(f'FIRE -> Precision: {precision[1]:.4f} - Recall: {recall[1]:.4f} - Accuracy: {accuracy[1]:.4f} - F1: {f1[1]:.4f}')
    
    return (
        {
        'Total': val_mean_loss, 
        'Smoke': smoke_mean_loss, 
        'Fire': fire_mean_loss
        },
        {
        'Accuracy': [accuracy[0].item(), accuracy[1].item()],
        'Precision': [precision[0].item(), precision[1].item()],
        'Recall': [recall[0].item(), recall[1].item()],
        'F1': [f1[0].item(), f1[1].item()] 
        }
    )