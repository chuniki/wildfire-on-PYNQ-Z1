from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import onnxruntime
import finn.core.onnx_exec as oxe


def get_label(label_file):

    '''
    Receives a txt file and returns the label associated, as [smoke?, fire?]
    '''

    label_array = np.zeros((2))
    
    with open(label_file) as f:
        lines = f.readlines()
        
        for line in lines:
            class_id, _, _, _, _ = line.strip().split()
            class_id = int(class_id)
            if np.array_equal(label_array, np.array([1, 1])):
                break
            else:
                label_array[class_id] = 1.
    
    return label_array    

def load_image_and_label(img_file, labels_dir, divide_255=True):

    '''
    Receives image and label files and returns the image ready for inference and corresponding label
    
    '''
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_LINEAR)
    if divide_255 == True:
        img = (img / 255).astype(np.float32)
    else:
        img = img.astype(np.float32)    
    img = np.expand_dims(img, axis=0)
    img = img.transpose(0, 3, 1, 2)

    img_name = Path(img_file).stem
    label_file = labels_dir + img_name + '.txt'
    label = get_label(label_file)

    return img, label

def get_metrics(y_pred, y_true, class_name=None):
    
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    n_samples = y_true.shape[0]
    #print(f'Number of test samples: {n_samples}')

    TP = (y_pred * y_true).sum()
    FP = (y_pred * (1 - y_true)).sum()
    TN = ((1 - y_pred) * (1 - y_true)).sum()
    FN = n_samples - TP - FP - TN
    assert FN == ((1 - y_pred) * y_true).sum()
    assert (TP + FP) != 0 # Avoid division by zero in Precision
    assert y_true.sum() != 0 # Avoid division by zero in Recall

    print(f'{class_name:<5} -> TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}')

    accuracy = (TP + TN) / n_samples 
    precision =  TP / (TP + FP)  
    recall = TP / y_true.sum()
    f1 = 2 * ( precision * recall ) / ( precision + recall )
    
    return {
        'Accuracy': round(accuracy, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1': round(f1, 4)
    }
    
def get_mean_metrics(smoke_metrics, fire_metrics):
    mean_dic = {}
    for s, f in zip(smoke_metrics.keys(), fire_metrics.keys()):
        mean_dic.update({s: round( ( smoke_metrics[s] + fire_metrics[f] ) / 2, 4)})
    return mean_dic


def eval_quant_onnx(imgs_list, labels_dir, model_wrapped, bipolar, divide_255):
   
    print(f'Number of test samples: {len(imgs_list)}\n')
        
    smoke_true = []
    smoke_pred = []
    fire_true = []
    fire_pred = []

    loop = tqdm(imgs_list, desc='Testing', leave=True)
  
    for img_file in loop:
    #for img_file in imgs_list:
  
        img, label = load_image_and_label(img_file, labels_dir, divide_255)

        input_dict = {"global_in": img}
        output_dict = oxe.execute_onnx(model_wrapped, input_dict)
        yhat = output_dict[list(output_dict.keys())[0]][0]

        if bipolar == False:
            # print(f'yhat before binarization: {yhat}')
            yhat[yhat > 0] = 1.
            yhat[yhat <= 0] = 0.
            # print(f'yhat after binarization: {yhat}')
        elif bipolar == True:
            # print(f'yhat before binarization: {yhat}')
            yhat[yhat < 1] = 0.
            # print(f'yhat after binarization: {yhat}')
        else:
            raise SystemExit("Wrong Output: neither bipolar nor fp32")
        # print(f'Image file: {img_file}: \n\tlabel {label}\n\tpred {yhat}') 
        smoke_pred.append(yhat[0])
        smoke_true.append(label[0])
        fire_pred.append(yhat[1])
        fire_true.append(label[1])

    # Custom Metrics
    smoke_metrics = get_metrics(smoke_pred, smoke_true, 'Smoke')
    fire_metrics = get_metrics(fire_pred, fire_true, 'Fire')
    mean_metrics = get_mean_metrics(smoke_metrics, fire_metrics)
    
    return {
        'Smoke': smoke_metrics,
        'Fire': fire_metrics,
        'Mean': mean_metrics
    }
