# set the configuration
import configparser
import os
import sys
from pathlib import Path

import cv2
import numpy as np

import torch
import torchvision

import seg_data



## Read the config
def read_ini(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)
    return config


assert len(sys.argv) ==2, "Please make sure you add the .ini file as a configuration"
assert os.path.isfile(sys.argv[1]), "Configuration file not exist"

config = read_ini(sys.argv[1])


img_path = config["DIR"]["image_dir"]
# mask_path =config["DIR"]["mask_dir"]
checkpoint_path = config["DIR"]["checkpoint_path"]
output_path = config["DIR"]["output_path"]

scale = 20

Path(output_path).mkdir(parents=True, exist_ok=True)

assert os.path.isfile(checkpoint_path), "Checkpoint file not exist"


print(f'''Predicting info:
        Reading images from: {img_path}
        Reading the checkpoint from: {checkpoint_path}
        Images will be save in output directory: {output_path}
''') 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'''System info:
        Using device: {device}
        CPU cores: {os.cpu_count()}
        GPU count: {torch.cuda.device_count()}
''')


dataset_pred = seg_data.segDataset(img_path = img_path,scale=scale,is_train=False)
data_loader_pred = torch.utils.data.DataLoader(dataset_pred, batch_size=1, shuffle=False)


model = torch.load(checkpoint_path)
model = model.to(device)
model.eval()

dataset_pred.img_path
dataset_pred.imgs

## Iterate through the images and save them to the directory.
for idx, (img,img_info) in enumerate(data_loader_pred):
    img_name = dataset_pred.imgs[idx]
        
    img = img.to(device)  
    out = model(img)
    
    if "deeplab" in checkpoint_path:
        out = out['out']
    out_temp = out.cpu().detach().numpy()



    seg= out_temp[0].transpose(1, 2, 0).argmax(2)
    
    # output_mask = np.zeros(seg.shape).astype('uint8')
    # output_mask[seg==1]=255

    if scale!=1:
        seg = cv2.resize(seg, (img_info['w'].item(),img_info['h'].item()),
                interpolation = cv2.INTER_NEAREST )
    # print(os.path.join(output_path,img_name))
    cv2.imwrite( os.path.join(output_path,img_name),seg)
    #  np.interp(img[0].cpu().detach().numpy().transpose(1, 2, 0),[0,1],[1,255]).astype('uint8'))
