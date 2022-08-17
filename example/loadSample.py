#%%
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

from config import TRAIN_CONFIG
from data.CustomDataset import HuBMAPDataset, get_aug

mean = np.array([0.7720342, 0.74582646, 0.76392896])
std = np.array([0.24745085, 0.26182273, 0.25782376])

ds = HuBMAPDataset(tfms=get_aug())
dl = DataLoader(ds,batch_size=64,shuffle=False,num_workers=TRAIN_CONFIG['NUM_WORKERS'])
imgs,masks = next(iter(dl))


plt.figure(figsize=(16,16))
for i,(img,mask) in enumerate(zip(imgs,masks)):
    img = ((img.permute(1,2,0)*std + mean)*255.0).numpy().astype(np.uint8)
    # print(i)
    plt.subplot(8,8,i+1)
    plt.imshow(img,vmin=0,vmax=255)
    plt.imshow(mask.squeeze().numpy(), alpha=0.2)
    plt.axis('off')
    plt.subplots_adjust(wspace=None, hspace=None)
    
del ds,dl,imgs,masks

#%%