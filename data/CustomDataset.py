from fastai.vision.all import *
from torch.utils.data import Dataset
from albumentations import *
import cv2
import pandas as pd
import numpy as np
from config import TRAIN_CONFIG
from sklearn.model_selection import KFold

mean = np.array([0.7720342, 0.74582646, 0.76392896])
std = np.array([0.24745085, 0.26182273, 0.25782376])


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


class HuBMAPDataset(Dataset):
    def __init__(self, fold=TRAIN_CONFIG['fold'], train=True, tfms=None):
        ids = pd.read_csv(TRAIN_CONFIG['LABELS']).id.astype(str).values
        kf = KFold(n_splits=TRAIN_CONFIG['nfolds'], random_state=TRAIN_CONFIG['SEED'], shuffle=True)
        ids = set(ids[list(kf.split(ids))[fold][0 if train else 1]])
        self.fnames = [fname for fname in os.listdir(
            TRAIN_CONFIG['TRAIN']) if fname.split('_')[0] in ids]
        self.train = train
        self.tfms = tfms

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = cv2.cvtColor(cv2.imread(
            os.path.join(TRAIN_CONFIG['TRAIN'], fname)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(TRAIN_CONFIG['MASKS'], fname), cv2.IMREAD_GRAYSCALE)
        if self.tfms is not None:
            augmented = self.tfms(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']
        return img2tensor((img/255.0 - mean)/std), img2tensor(mask)


def get_aug(p=1.0):
    return Compose([
        HorizontalFlip(),
        VerticalFlip(),
        RandomRotate90(),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.9,
                         border_mode=cv2.BORDER_REFLECT),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.3),
        OneOf([
            HueSaturationValue(10, 15, 10),
            CLAHE(clip_limit=2),
            RandomBrightnessContrast(),
        ], p=0.3),
    ], p=p)
