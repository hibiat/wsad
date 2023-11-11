# Feature extractor from imagenet pretrained model


import os
import numpy as np
import timm
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import glob
from PIL import Image

modelnames = ['vit_large_patch16_224_in21k'] #'efficientnet_b4', 'tf_efficientnet_b4', 'vit_base_patch16_224_in21k'
rootdir = '/path_to/rsna-intracranial-hemorrhage-detection/'
in_imgdirs = [rootdir+'stage_1_test_png_1ch_mask', rootdir+'stage_1_train_png_1ch_mask', \
            rootdir+'stage_1_test_png_3ch_mask', rootdir+'stage_1_train_png_3ch_mask']

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

for in_imgdir in in_imgdirs:
    out_dir = in_imgdir + '_feature'
    for modelname in modelnames:
        out_dir_name = os.path.join(out_dir, modelname)
        if not os.path.exists(out_dir_name):
            os.makedirs(out_dir_name)
        model = timm.create_model(modelname, pretrained=True, num_classes=0)
        #Get input size, mean, and std
        modelinfo = model.default_cfg
        mean = modelinfo['mean']
        std = modelinfo['std']
        input_size = modelinfo['input_size']
        input_h = input_size[1]
        input_w = input_size[2]
        transform = transforms.Compose([transforms.ConvertImageDtype(torch.float),\
                                transforms.Normalize(mean=mean, std=std), \
                                transforms.Resize((input_h, input_w))
                                ])   
        with tqdm(total=len(os.listdir(in_imgdir)), postfix=os.path.basename(os.path.dirname(in_imgdir+'/'))+'('+modelname+')') as pbar:
            for studyid in os.listdir(in_imgdir):
                if os.path.isdir(os.path.join(in_imgdir, studyid)):
                    numslice = len(glob.glob(os.path.join(in_imgdir, studyid, '*.png')))
                    img_list = []
                    for slice in range(numslice):
                        filename = os.path.join(in_imgdir, studyid, studyid+'_'+str(slice).zfill(3)+'.png')
                        img = Image.open(filename)
                        img_array = np.array(img)
                        if img_array.ndim == 2: # grayscale image is converted to 3ch
                            img_array = np.concatenate([img_array[:,:,np.newaxis], img_array[:,:,np.newaxis], img_array[:,:,np.newaxis]], axis=2)
                        img_tensor = torch.from_numpy(img_array).permute(2,0,1)
                        img_tensor = transform(img_tensor)
                        img_tensor = img_tensor.unsqueeze(0) #shape:[1,3,h,w]
                        img_list.append(img_tensor)
                    img_tensors = torch.cat(img_list, dim=0) #shape:[numslice,3,h,w]
                    img_tensors.to(device)
                    feature = model(img_tensors).data.cpu().numpy()
                    np.save(os.path.join(out_dir_name, studyid+'.npy'), feature)
                    
                pbar.update(1)








