import sys
import matplotlib.pyplot as plt
import pandas as pd
import torch
import config
if config.BREVITAS_MODEL == True:
    from brevitas.export import export_onnx_qcdq
    from brevitas.export import export_qonnx

class LogMetrics():
    
    def __init__(self):
        
        self.smoke_accuracy = []
        self.smoke_precision = []
        self.smoke_recall = []
        self.smoke_f1 = []
        self.fire_accuracy = []
        self.fire_precision = []
        self.fire_recall = []
        self.fire_f1 = []
        self.mean_accuracy = []
        self.mean_precision = []
        self.mean_recall = []
        self.mean_f1 = []
        
        self.smoke_metrics_dic = {
            'Accuracy': self.smoke_accuracy,
            'Precision': self.smoke_precision,
            'Recall': self.smoke_recall,
            'F1': self.smoke_f1
        }
        self.fire_metrics_dic = {
            'Accuracy': self.fire_accuracy,
            'Precision': self.fire_precision,
            'Recall': self.fire_recall,
            'F1': self.fire_f1
        }
        self.mean_metrics_dic = {
            'Accuracy': self.mean_accuracy,
            'Precision': self.mean_precision,
            'Recall': self.mean_recall,
            'F1': self.mean_f1
        }
                
    def update_metrics(self, metrics):
        for k, v in metrics.items():
            self.smoke_metrics_dic[k].append(v[0])
            self.fire_metrics_dic[k].append(v[1])
            self.mean_metrics_dic[k].append( ( v[0] + v[1] ) / 2 )
            
    def get_metrics(self):
        return {
            'Smoke': self.smoke_metrics_dic, 
            'Fire': self.fire_metrics_dic, 
            'Mean': self.mean_metrics_dic
        }
    
    def save_to_csv(self, save_path):
        full_dic = {
            'Smoke': self.smoke_metrics_dic,
            'Fire': self.fire_metrics_dic,
            'Mean': self.mean_metrics_dic
        }
        n_metrics = len(self.smoke_metrics_dic)
        n_classes = len(full_dic)
        n_epochs = len(self.smoke_metrics_dic['F1'])
        
        data = np.zeros((n_epochs, n_metrics*n_classes))
        
        for i, kout in enumerate(full_dic.keys()):
            for j, kin in enumerate(full_dic[kout].keys()):
                for epoch in range(n_epochs):
                    data[epoch, j*n_classes+i] = full_dic[kout][kin][epoch]
        
        cols = pd.MultiIndex.from_product([['Accuracy', 'Precision', 'Recall', 'F1'], ['Smoke', 'Fire', 'Mean']])
        df_big = pd.DataFrame(data, columns=cols)
        df_big.to_csv(save_path)


class LogLosses():

    def __init__(self):
        self.total = []
        self.smoke = []
        self.fire = []
        self.losses = {
            'Total': self.total,
            'Smoke': self.smoke,
            'Fire': self.fire,
        }
                
    def update_metrics(self, metrics):
        for k, v in metrics.items():
            self.losses[k].append(v)
            
    def get_metrics(self):
        return self.losses

class LogLR():

    def __init__(self, log_path):
        self.log_path = log_path
        self.lr = []
                
    def log_lr(self, lr):
        self.lr.append(lr)
            
    def plot_lr(self, epochs):
        plt.plot(epochs, self.lr)
        plt.title("Learning Rate")
        plt.xlabel("Epochs")
        plt.ylabel("Learning Rate")
        plt.grid()
        plt.savefig(self.log_path + "learning_rate.png")
        plt.close()    
    

class PlotMetrics():
    def __init__(self, log_path, model_name, loss_or_metric):
        self.log_path = log_path
        self.model_name = model_name
        self.loss_or_metric = loss_or_metric
    
    def __plot_one_metric__(
        self,
        smoke_or_fire: str, 
        metric_name: str, 
        train_metric: list, 
        val_metric: list, 
        epochs: list):
        '''
        Plot one metric including Train and Val data
        '''
        plt.plot(epochs, train_metric, label = "Train" + f' {smoke_or_fire}' + f' {metric_name}')
        plt.plot(epochs, val_metric, label = "Val" + f' {smoke_or_fire}' + f' {metric_name}')
        plt.title("Train vs Val:" + f' {smoke_or_fire}' + f' {metric_name}')
        plt.xlabel("Epochs")
        plt.ylabel(metric_name)
        if metric_name != 'Loss':
            plt.ylim([0,1])
        plt.legend()
        plt.grid()
        plt.savefig(self.log_path + f'{self.model_name}_' + f'_{smoke_or_fire}_' + f'_{metric_name}' + ".png")
        plt.close()    

    def plot_all_metrics(self, train_metrics, val_metrics, epochs):
        
        # Plot Losses
        if self.loss_or_metric == "Loss":
            for (key_train, value_train), (key_val, value_val) in zip(train_metrics.items(), val_metrics.items()):
                self.__plot_one_metric__(
                    smoke_or_fire = key_train, 
                    metric_name = 'Loss', 
                    train_metric = value_train, 
                    val_metric = value_val, 
                    epochs = epochs)

        elif self.loss_or_metric == "Metric":
            for k_train, k_val in zip(train_metrics.keys(), val_metrics.keys()):
                for (key_train, value_train), (key_val, value_val) in zip(train_metrics[k_train].items(), val_metrics[k_val].items()):
                    self.__plot_one_metric__(
                        smoke_or_fire = k_train, 
                        metric_name = key_train, 
                        train_metric = value_train, 
                        val_metric = value_val, 
                        epochs = epochs)
                    
        else:
            raise SystemExit("Wrong Loss or Metric request to Plot")
    
def print_metrics_to_logger(train_or_val, losses, metrics):
    str_metric = 'Total Loss'.ljust(11) + '|Smoke Loss'.ljust(11) + '|Fire Loss'.ljust(11) + '|    '
    str_smoke = '-----------'.ljust(11) + '|----------'.ljust(11) + '|----------'.ljust(11) + '|    '
    str_fire = (f'{losses["Total"]:.2f}'.ljust(11) + 
                f'|{losses["Smoke"]:.2f}'.ljust(11) + 
                f'|{losses["Fire"]:.2f}'.ljust(11) + '|    ')

    str_metric += '_______'.ljust(7)
    str_smoke += 'Smoke'.ljust(7)
    str_fire += 'Fire'.ljust(7)
    for k in metrics.keys():
        str_metric += f'|{k}'.ljust(10)
        str_smoke += f'|{metrics[k][0]:.4f}'.ljust(10)
        str_fire += f'|{metrics[k][1]:.4f}'.ljust(10)
    return f'{train_or_val}\n{str_metric}|\n{str_smoke}|\n{str_fire}|\n'

def save_checkpoint(epoch, model, optimizer, scheduler, checkpoint_name):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, 
    checkpoint_name) 
    
def load_checkpoint(model_path, model, optimizer=None, scheduler=None, device='cpu'):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"Loading Model. Trained during {start_epoch} epochs")
    return start_epoch

def export_onnx(model, input_shape, filename, device):
    model.eval()
    model.cpu()
    export_qonnx(model, torch.randn(input_shape), filename + '__QONNX.onnx')
    #export_onnx_qcdq(model, torch.randn(input_shape), filename + '__QCDQ.onnx')
    model.to(device)
    print(f'Model exported to ONNX: {filename}')


if __name__ == "__main__":
    folder = './zzz_test_metrics&loss_plots/'

    train_metrics_logger = LogMetrics()
    train_losses_logger = LogLosses()
    val_metrics_logger = LogMetrics()
    val_losses_logger = LogLosses()
    
    loss_plotter = PlotMetrics(log_path=folder, model_name='BED', loss_or_metric='Loss')
    metrics_plotter = PlotMetrics(log_path=folder, model_name='BED', loss_or_metric='Metric')
    
    epochs = [0, 1, 2, 3, 4]
    
    train_loss_dic = {
        'Total': [50, 40, 30, 20, 10],
        'Smoke': [47, 37, 27, 17, 7],
        'Fire': [5, 4, 3, 2, 1]
    }
    val_loss_dic = {
        'Total': [52, 42, 32, 22, 12],
        'Smoke': [47.8, 37.9, 27.8, 17.9, 7.5],
        'Fire': [7, 6, 6, 5, 4]
    }
    
    train_metrics_dic = {
        'Accuracy': [[0.4, 0.5, 0.6, 0.7, 0.8], [0.42, 0.52, 0.62, 0.72, 0.82]],
        'Precision': [[0.34, 0.35, 0.36, 0.37, 0.38], [0.342, 0.352, 0.362, 0.372, 0.382]],
        'Recall': [[0.2, 0.3, 0.4, 0.5, 0.6], [0.25, 0.35, 0.45, 0.55, 0.65]],
        'F1': [[0.14, 0.15, 0.16, 0.17, 0.18], [0.242, 0.352, 0.462, 0.572, 0.682]]
        }
    val_metrics_dic = {
        'Accuracy': [[0.3, 0.4, 0.5, 0.6, 0.7], [0.429, 0.529, 0.629, 0.729, 0.829]],
        'Precision': [[0.24, 0.25, 0.26, 0.27, 0.38], [0.341, 0.351, 0.361, 0.371, 0.381]],
        'Recall': [[0.2, 0.35, 0.4, 0.55, 0.6], [0.125, 0.135, 0.145, 0.155, 0.165]],
        'F1': [[0.14, 0.55, 0.16, 0.77, 0.18], [0.442, 0.352, 0.462, 0.572, 0.682]]
        }
    
    # Loop to update metrics emulating training loop
    for i in range(len(epochs)):
        train_losses_logger.update_metrics({
            'Total': train_loss_dic['Total'][i],
            'Smoke': train_loss_dic['Smoke'][i],
            'Fire': train_loss_dic['Fire'][i],
        }) 
        val_losses_logger.update_metrics({
            'Total': val_loss_dic['Total'][i],
            'Smoke': val_loss_dic['Smoke'][i],
            'Fire': val_loss_dic['Fire'][i],
        })
        train_metrics_logger.update_metrics({
            'Accuracy': [train_metrics_dic['Accuracy'][0][i], train_metrics_dic['Accuracy'][1][i]], 
            'Precision': [train_metrics_dic['Precision'][0][i], train_metrics_dic['Precision'][1][i]], 
            'Recall': [train_metrics_dic['Recall'][0][i], train_metrics_dic['Recall'][1][i]], 
            'F1': [train_metrics_dic['F1'][0][i], train_metrics_dic['F1'][1][i]], 
        })
        val_metrics_logger.update_metrics({
            'Accuracy': [val_metrics_dic['Accuracy'][0][i], val_metrics_dic['Accuracy'][1][i]], 
            'Precision': [val_metrics_dic['Precision'][0][i], val_metrics_dic['Precision'][1][i]], 
            'Recall': [val_metrics_dic['Recall'][0][i], val_metrics_dic['Recall'][1][i]], 
            'F1': [val_metrics_dic['F1'][0][i], val_metrics_dic['F1'][1][i]], 
        })                
        print(f'Train Losses \n{train_losses_logger.get_metrics()}')
        print(f'Val Losses \n{val_losses_logger.get_metrics()}')
        print(f'Train Metrics \n{train_metrics_logger.get_metrics()}')
        print(f'Val Metrics \n{val_metrics_logger.get_metrics()}')
        

    loss_plotter.plot_all_metrics(
        train_losses_logger.get_metrics(),
        val_losses_logger.get_metrics(),
        epochs)

    metrics_plotter.plot_all_metrics(
        train_metrics_logger.get_metrics(),
        val_metrics_logger.get_metrics(),
        epochs)
    

    


    
    