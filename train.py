import gc
import os

from model.evaluate import Dice_th_pred, Model_pred, save_img
# from model.unext50 import UneXt50, split_layers
# from model.unexteffb4 import UneXt50, split_layers
from model.unexteff_v2l import UneXt50, split_layers
from data.CustomDataset import HuBMAPDataset, get_aug
from fastai.vision.all import *
from util.lossfunc import symmetric_lovasz, Dice_soft, Dice_th
from config import TRAIN_CONFIG

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]="1"
# export PYTORCH_ENABLE_MPS_FALLBACK=1
dice = Dice_th_pred(np.arange(0.2,0.7,0.01))
for fold in range(TRAIN_CONFIG['nfolds']):
    ds_t = HuBMAPDataset(fold=fold, train=True, tfms=get_aug())
    ds_v = HuBMAPDataset(fold=fold, train=False)
    data = ImageDataLoaders.from_dsets(ds_t,ds_v,bs=TRAIN_CONFIG['bs'],
                num_workers=TRAIN_CONFIG['NUM_WORKERS'],pin_memory=True).to(TRAIN_CONFIG['DEVICE'])
    model = UneXt50().to(TRAIN_CONFIG['DEVICE'])
    
    learn = Learner(data, model, loss_func=symmetric_lovasz,
                metrics=[Dice_soft(),Dice_th()], 
                splitter=split_layers).to_fp16()
    
    #start with training the head
    learn.freeze_to(-1) #doesn't work
    for param in learn.opt.param_groups[0]['params']:
        param.requires_grad = False
    learn.fit_one_cycle(TRAIN_CONFIG["freeze_epoch"], lr_max=0.5e-2)

    #continue training full model
    learn.unfreeze()
    learn.fit_one_cycle(TRAIN_CONFIG["unfreeze_epoch"], lr_max=slice(2e-4,2e-3),
        cbs=SaveModelCallback(monitor='dice_th',comp=np.greater))
    torch.save(learn.model.state_dict(),f'model_{fold}.pth')
    
    #model evaluation on val and saving the masks
    mp = Model_pred(learn.model,learn.dls.loaders[1])
    with zipfile.ZipFile('val_masks_tta.zip', 'a') as out:
        for p in progress_bar(mp):
            dice.accumulate(p[0],p[1])
            save_img(p[0],p[2],out)
    gc.collect()