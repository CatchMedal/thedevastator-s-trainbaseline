import gc, os, sys, getopt
from GPUtil import showUtilization as gpu_usage
import wandb

from model.evaluate import Dice_th_pred, Model_pred, save_img
# from model.unext50 import UneXt50, split_layers
# from model.unexteffb4 import UneXt50, split_layers
# from model.unexteff_v2l import UneXt50, split_layers
from model.unexteff_b7 import UneXt50, split_layers
from data.CustomDataset import HuBMAPDataset, get_aug
from fastai.vision.all import *
from fastai.basics import Callback
from fastai.callback.wandb import *
from util.lossfunc import symmetric_lovasz, Dice_soft, Dice_th
from config import TRAIN_CONFIG


def CleanGPU(t):

    gpu_usage()                             

    gc.collect()
    torch.cuda.empty_cache()

    print("GPU Usage after emptying the cache")
    gpu_usage()


argv = sys.argv
opts, etc_args = getopt.getopt(argv[1:],"e:",["env="])
learnler_cbs=[Callback(after_epoch=CleanGPU)]
for opt, arg in opts:
    if arg == 'kaggle':
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        wandb_api = user_secrets.get_secret("wandb_api") 
        wandb.login(key=wandb_api)
        wandb.init(project="hubmap-unext", entity="mglee_")
        learnler_cbs.append(WandbCallback())
    elif arg == 'colab': #Colab, GCP, Etc,,,
        wandb.login()
        learnler_cbs.append(WandbCallback())
        wandb.init(project="hubmap-unext", entity="mglee_")
    else: #For Local ENV
        pass

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
                splitter=split_layers, cbs=learnler_cbs).to_fp16()
    
    #memory optimization code
    #start with training the head
    with learn.no_bar():
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
            wandb.log({'p0':p[0], 'p1':p[1]})
    gc.collect()

