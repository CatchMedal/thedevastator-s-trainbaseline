TRAIN_CONFIG = {
    'bs': 32,
    'nfolds': 4,
    'fold': 0,
    'SEED': 2021,
    'TRAIN': 'input/hubmap-2022-256x256/train/',
    'MASKS': 'input/hubmap-2022-256x256/masks/',
    'LABELS': 'input/hubmap-organ-segmentation/train.csv',
    'NUM_WORKERS': 0,
    'ENABLE_CUDA':False,
    "DEVICE":"mps",
    "freeze_epoch":4,
    "unfreeze_epoch":15
}