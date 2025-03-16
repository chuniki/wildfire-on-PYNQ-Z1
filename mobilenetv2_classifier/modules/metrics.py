import torchmetrics
import config

######################################################################################
#                                       CUDA                                         #
######################################################################################
precision_metric = torchmetrics.classification.MultilabelPrecision(num_labels = config.NUM_CLASSES, 
                                                                   threshold = 0.5, 
                                                                   average = None).to(config.DEVICE)
recall_metric = torchmetrics.classification.MultilabelRecall(num_labels = config.NUM_CLASSES, 
                                                             threshold = 0.5, 
                                                             average = None).to(config.DEVICE)
accuracy_metric = torchmetrics.classification.MultilabelAccuracy(num_labels = config.NUM_CLASSES, 
                                                                 threshold = 0.5, 
                                                                 average = None).to(config.DEVICE)
f1_metric = torchmetrics.classification.MultilabelF1Score(num_labels = config.NUM_CLASSES, 
                                                          threshold = 0.5, 
                                                          average = None).to(config.DEVICE)

f1_metric_mean = torchmetrics.classification.MultilabelF1Score(num_labels = config.NUM_CLASSES, 
                                                               threshold = 0.5, 
                                                               average = 'macro').to(config.DEVICE)

######################################################################################
#                                       CPU                                          #
######################################################################################
precision_metric_cpu = torchmetrics.classification.MultilabelPrecision(num_labels = config.NUM_CLASSES, 
                                                                   threshold = 0.5, 
                                                                   average = None).to('cpu')
recall_metric_cpu = torchmetrics.classification.MultilabelRecall(num_labels = config.NUM_CLASSES, 
                                                             threshold = 0.5, 
                                                             average = None).to('cpu')
accuracy_metric_cpu = torchmetrics.classification.MultilabelAccuracy(num_labels = config.NUM_CLASSES, 
                                                                 threshold = 0.5, 
                                                                 average = None).to('cpu')
f1_metric_cpu = torchmetrics.classification.MultilabelF1Score(num_labels = config.NUM_CLASSES, 
                                                          threshold = 0.5, 
                                                          average = None).to('cpu')

f1_metric_mean_cpu = torchmetrics.classification.MultilabelF1Score(num_labels = config.NUM_CLASSES, 
                                                               threshold = 0.5, 
                                                               average = 'macro').to('cpu')

