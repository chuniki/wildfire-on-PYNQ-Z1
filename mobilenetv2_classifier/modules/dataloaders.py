import config
import modules.dataset_dfire as dataset_dfire
import modules.dataset_fasdd as dataset_fasdd
import modules.dataset_clouds as dataset_clouds
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_loader(add_clouds=False):
    train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(p=0.2),
                A.HueSaturationValue(hue_shift_limit=10, p=0.2),
                A.Blur(blur_limit=(17,17), p=0.3),
                A.CLAHE(clip_limit=2.0, p=0.2),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.1),
            ], p=0.9),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, p=0.3),
            A.Resize(config.IMG_H, config.IMG_W, p=1),
            ToTensorV2(p=1),
        ]
    )
    
    # TRAIN DATASET
    print("\n====================\nTRAIN DFIRE dataset")
    train_dfire_dataset = dataset_dfire.DFireDataset(
        img_h = config.IMG_H,
        img_w = config.IMG_W,
        img_dir = config.DFIRE_TRAIN_IMG_DIR,
        label_dir = config.DFIRE_TRAIN_LABEL_DIR,
        num_classes=config.NUM_CLASSES,
        ds_len = config.DS_LEN,
        transform=train_transform)
    print(f'\nTrain DFire dataset len: {len(train_dfire_dataset)}')

    print("\n====================\nTRAIN FASDD UAV dataset")
    train_fasdd_uav_ds = dataset_fasdd.FASDDDataset(
        img_h=config.IMG_H, 
        img_w=config.IMG_W, 
        imgs_dir=config.FASDD_UAV_IMGS_DIR, 
        labels_file=config.FASDD_UAV_TRAIN_LABELS_FILE, 
        num_classes=config.NUM_CLASSES,
        ds_len=config.DS_LEN,
        transform=train_transform)
    print(f'\nTrain FASDD UAV dataset len: {len(train_fasdd_uav_ds)}')

    print("\n====================\nVAL FASDD UAV dataset")
    val_fasdd_uav_ds = dataset_fasdd.FASDDDataset(
        img_h=config.IMG_H, 
        img_w=config.IMG_W, 
        imgs_dir=config.FASDD_UAV_IMGS_DIR, 
        labels_file=config.FASDD_UAV_VAL_LABELS_FILE, 
        num_classes=config.NUM_CLASSES,
        ds_len=config.DS_LEN,
        transform=train_transform)
    print(f'\nVal FASDD UAV dataset len: {len(val_fasdd_uav_ds)}')
    
    print("\n====================\nTRAIN FASDD CV dataset")
    train_fasdd_cv_ds = dataset_fasdd.FASDDDataset(
        img_h=config.IMG_H, 
        img_w=config.IMG_W, 
        imgs_dir=config.FASDD_CV_IMGS_DIR, 
        labels_file=config.FASDD_CV_TRAIN_LABELS_FILE, 
        num_classes=config.NUM_CLASSES,
        ds_len=config.DS_LEN,
        transform=train_transform)
    print(f'\nTrain FASDD CV dataset len: {len(train_fasdd_cv_ds)}')

    print("\n====================\nVal FASDD CV dataset")
    val_fasdd_cv_ds = dataset_fasdd.FASDDDataset(
        img_h=config.IMG_H, 
        img_w=config.IMG_W, 
        imgs_dir=config.FASDD_CV_IMGS_DIR, 
        labels_file=config.FASDD_CV_VAL_LABELS_FILE, 
        num_classes=config.NUM_CLASSES,
        ds_len=config.DS_LEN,
        transform=train_transform)
    print(f'\nVal FASDD CV dataset len: {len(val_fasdd_cv_ds)}')

    if add_clouds:
        print("\n====================\nTrain CLOUDS dataset")
        train_clouds_ds = dataset_clouds.Clouds(
            img_h=config.IMG_H, 
            img_w=config.IMG_W, 
            img_dir=config.CLOUDS_IMG_DIR, 
            num_classes=config.NUM_CLASSES,
            ds_len=config.DS_LEN,
            transform=train_transform)
        print(f'\nTrain CLOUDS dataset len: {len(train_clouds_ds)}')

    print("\n==================== CONCATENATE")
    print("\nConcatenate Train DFire and Train FASDD UAV datasets")
    train_ds = torch.utils.data.ConcatDataset((train_dfire_dataset, train_fasdd_uav_ds))
    print(f'Train dataset len: {len(train_ds)}')
    print("Concatenate with Val FASDD UAV dataset")
    train_ds = torch.utils.data.ConcatDataset((train_ds, val_fasdd_uav_ds))
    print(f'Train dataset len: {len(train_ds)}')
    print("Concatenate with Train FASDD CV dataset")
    train_ds = torch.utils.data.ConcatDataset((train_ds, train_fasdd_cv_ds))
    print(f'Train dataset len: {len(train_ds)}')
    print("Concatenate with Val FASDD CV dataset")
    train_ds = torch.utils.data.ConcatDataset((train_ds, val_fasdd_cv_ds))
    print(f'Train dataset len: {len(train_ds)}')
    if add_clouds:
        print("Concatenate with Train Clouds dataset")
        train_ds = torch.utils.data.ConcatDataset((train_ds, train_clouds_ds))
        print(f'Train dataset len: {len(train_ds)}')
    
    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,
        drop_last=True)
    
    return train_loader

def get_val_loader(return_img_filename=False):
    val_transform = A.Compose([
        A.Resize(config.IMG_H, config.IMG_W, p=1),
        ToTensorV2(p=1),
        ]
    )
    
    print("\n====================\nTEST DFire dataset")
    val_dfire_dataset = dataset_dfire.DFireDataset(
        img_h = config.IMG_H,
        img_w = config.IMG_W,
        img_dir = config.DFIRE_TEST_IMG_DIR,
        label_dir = config.DFIRE_TEST_LABEL_DIR,
        num_classes = config.NUM_CLASSES,
        ds_len = config.DS_LEN,
        transform=val_transform,
        return_img_filename=return_img_filename)
    print(f'\nTest dataset len: {len(val_dfire_dataset)}')
    
    print("\n====================\nTEST FASDD UAV dataset")
    val_fasdd_uav_ds = dataset_fasdd.FASDDDataset(
        img_h=config.IMG_H, 
        img_w=config.IMG_W, 
        imgs_dir=config.FASDD_UAV_IMGS_DIR, 
        labels_file=config.FASDD_UAV_TEST_LABELS_FILE, 
        num_classes=config.NUM_CLASSES,
        ds_len=config.DS_LEN,
        transform=val_transform,
        return_img_filename=return_img_filename)
    print(f'\nTest FASDD UAV dataset len: {len(val_fasdd_uav_ds)}')
    
    print("\n====================\nTEST FASDD CV dataset")
    val_fasdd_cv_ds = dataset_fasdd.FASDDDataset(
        img_h=config.IMG_H, 
        img_w=config.IMG_W, 
        imgs_dir=config.FASDD_CV_IMGS_DIR, 
        labels_file=config.FASDD_CV_TEST_LABELS_FILE, 
        num_classes=config.NUM_CLASSES,
        ds_len=config.DS_LEN,
        transform=val_transform,
        return_img_filename=return_img_filename)
    print(f'\nTest FASDD CV dataset len: {len(val_fasdd_cv_ds)}')

    print("\n==================== CONCATENATE")
    print("Concatenate Test DFire and FASDD UAV datasets")
    val_ds_concat = torch.utils.data.ConcatDataset((val_dfire_dataset, val_fasdd_uav_ds))
    print(f'Test dataset len: {len(val_ds_concat)}')
    print("Concatenate with FASDD CV dataset")
    val_ds = torch.utils.data.ConcatDataset((val_ds_concat, val_fasdd_cv_ds))
    print(f'Test dataset len: {len(val_ds)}')
    
    val_loader = DataLoader(
        dataset=val_ds,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=True)
    
    return val_loader


# ______________________________________________________________________ #
#                         DFIRE TEST DATASETS                            #
# ______________________________________________________________________ #
def get_dfire_val_loader(return_img_filename=False):
    val_transform = A.Compose([
        A.Resize(config.IMG_H, config.IMG_W, p=1),
        ToTensorV2(p=1),
        ]
    )
    
    print("\n====================\nTEST DFire dataset")
    val_dfire_dataset = dataset_dfire.DFireDataset(
        img_h = config.IMG_H,
        img_w = config.IMG_W,
        img_dir = config.DFIRE_TEST_IMG_DIR,
        label_dir = config.DFIRE_TEST_LABEL_DIR,
        num_classes = config.NUM_CLASSES,
        ds_len = config.DS_LEN,
        transform=val_transform,
        return_img_filename=return_img_filename)
    print(f'\nTest dataset len: {len(val_dfire_dataset)}')

    val_loader = DataLoader(
        dataset=val_dfire_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=False)
    
    return val_loader


# ______________________________________________________________________ #
#                         DFIRE MINI DATASETS                            #
# ______________________________________________________________________ #

def get_dfire_mini_train_loader():
    val_transform = A.Compose([
        A.Resize(config.IMG_H, config.IMG_W, p=1),
        ToTensorV2(p=1),
        ]
    )
    
    print("\n====================\nTRAIN DFire MINI dataset")
    val_ds = dataset_dfire.DFireDataset(
        img_h = config.IMG_H,
        img_w = config.IMG_W,
        img_dir = config.DFIRE_MINI_TRAIN_IMG_DIR,
        label_dir = config.DFIRE_MINI_TRAIN_LABEL_DIR,
        num_classes = config.NUM_CLASSES,
        ds_len = config.DS_LEN,
        transform=val_transform)
    print(f'\nTest dataset len: {len(val_ds)}')
    
    val_loader = DataLoader(
        dataset=val_ds,
        batch_size=1,
        num_workers=1,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=False)
    
    return val_loader

def get_dfire_mini_test_loader():
    val_transform = A.Compose([
        A.Resize(config.IMG_H, config.IMG_W, p=1),
        ToTensorV2(p=1),
        ]
    )
    
    print("\n====================\nTEST DFire MINI dataset")
    val_ds = dataset_dfire.DFireDataset(
        img_h = config.IMG_H,
        img_w = config.IMG_W,
        img_dir = config.DFIRE_MINI_TEST_IMG_DIR,
        label_dir = config.DFIRE_MINI_TEST_LABEL_DIR,
        num_classes = config.NUM_CLASSES,
        ds_len = config.DS_LEN,
        transform=val_transform)
    print(f'\nTest dataset len: {len(val_ds)}')
    
    val_loader = DataLoader(
        dataset=val_ds,
        batch_size=1,
        num_workers=1,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=False)
    
    return val_loader

def get_clouds_loader(return_img_filename=False):
    val_transform = A.Compose([
        A.Resize(config.IMG_H, config.IMG_W, p=1),
        ToTensorV2(p=1),
        ]
    )
    
    print("\n====================\nTrain CLOUDS dataset")
    train_clouds_ds = dataset_clouds.Clouds(
        img_h=config.IMG_H, 
        img_w=config.IMG_W, 
        img_dir=config.CLOUDS_IMG_DIR, 
        num_classes=config.NUM_CLASSES,
        ds_len=config.DS_LEN,
        transform=val_transform,
        return_img_filename=return_img_filename)
    print(f'\nTrain CLOUDS dataset len: {len(train_clouds_ds)}')
    
    val_loader = DataLoader(
        dataset=train_clouds_ds,
        batch_size=1,
        num_workers=1,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=False)
    
    return val_loader

