from tqdm import tqdm
import modules.metrics as metrics
import torch

def eval_fn(loader, model, device):
    
    model.eval()
    loop = tqdm(loader, desc='Validating', leave=True)

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(device), y.to(device)
        out = model(x)   

        #print(f'Label {y} - Pred {out}')
        yhat = torch.sigmoid(out.detach())
        metrics.precision_metric_cpu.update(yhat, y)
        metrics.recall_metric_cpu.update(yhat, y)
        metrics.accuracy_metric_cpu.update(yhat, y)
        metrics.f1_metric_cpu.update(yhat, y)
    
    precision = metrics.precision_metric_cpu.compute()
    recall = metrics.recall_metric_cpu.compute()
    accuracy = metrics.accuracy_metric_cpu.compute()
    f1 = metrics.f1_metric_cpu.compute()
    
    metrics.precision_metric_cpu.reset()
    metrics.recall_metric_cpu.reset()
    metrics.accuracy_metric_cpu.reset()
    metrics.f1_metric_cpu.reset()

    print(f'SMOKE -> Precision: {precision[0]:.4f} - Recall: {recall[0]:.4f} - Accuracy: {accuracy[0]:.4f} - F1: {f1[0]:.4f}')
    print(f'FIRE -> Precision: {precision[1]:.4f} - Recall: {recall[1]:.4f} - Accuracy: {accuracy[1]:.4f} - F1: {f1[1]:.4f}')
    
    return {
        'Accuracy': [accuracy[0].item(), accuracy[1].item()],
        'Precision': [precision[0].item(), precision[1].item()],
        'Recall': [recall[0].item(), recall[1].item()],
        'F1': [f1[0].item(), f1[1].item()] 
        }
    