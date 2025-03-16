from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

def get_dfire_label(label_file):

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

def load_dfire_image_and_label(img_file, labels_dir):

    '''
    Receives image and label files and returns the image ready for inference and corresponding label
    
    '''
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_LINEAR)
    assert img.dtype == "uint8", "Image datatype must be UINT8"
    img = np.expand_dims(img, axis=0)

    img_name = Path(img_file).stem
    label_file = labels_dir + img_name + '.txt'
    label = get_dfire_label(label_file)

    return img, label

def load_fasdd_image_and_label(img_file):

    '''
    Receives image filename and returns the image ready for inference and corresponding label
    
    '''
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_LINEAR)
    assert img.dtype == "uint8", "Image datatype must be UINT8"
    img = np.expand_dims(img, axis=0)

    img_name = Path(img_file).stem
    if "both" in img_name:
        label = np.array([1., 1.])
    elif "neither" in img_name:
        label = np.array([0., 0.])
    elif "fire" in img_name:
        label = np.array([0., 1.])
    elif "smoke" in img_name:
        label = np.array([1., 0.]) 
    else:
        raise Exception('Wrong label in FASDD')
        
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


# def eval_accel(imgs_list, labels_dir, accel):
def eval_accel(datasets_dict, dfire_labels_dir, accel):
        
    smoke_true = []
    smoke_pred = []
    fire_true = []
    fire_pred = []

    for dataset, imgs_list in datasets_dict.items():
        
        print(f'Number of test samples {dataset}: {len(imgs_list)}\n')
        loop = tqdm(imgs_list, desc='Testing', leave=True)

        for img_file in loop:

            # print(f'Image file: {img_file}')
            if "dfire" in dataset:
                img, label = load_dfire_image_and_label(img_file, dfire_labels_dir)
            elif "fasdd" in dataset:
                img, label = load_fasdd_image_and_label(img_file)
            else:
                raise Exception('Wrong Dataset Name') 

            # Execute accelerator
            ibuf_normal = []
            ibuf_normal.append(img)
            obuf_normal = accel.execute(ibuf_normal)
            # if not isinstance(obuf_normal, list):
            #     obuf_normal = [obuf_normal]

            yhat = obuf_normal[0, :]

            # print(f'Image file: {img_file}: \n\tlabel {label}\n\tpred {yhat}') 
            # print(f'\tlabel {label}\n\tpred {yhat}') 
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
