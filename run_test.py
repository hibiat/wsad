import os
import datetime
import torch
from torch.utils.data import DataLoader
import glob


import parameters
from utils import checkfeaturesize
from model import model_generater
from dataset import dataset
from test import test, test_no_slicelabel, eval_ichsubtype

if __name__ == '__main__':
    args = parameters.parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda:{}".format(args.device))
    torch.cuda.set_device(args.device)
    feature_size = checkfeaturesize(args)
    model = model_generater(model_name=args.model_name, feature_size=feature_size).to(device)
    modelfilename = glob.glob(os.path.join(args.weightdir, '*.pth'))[0] 
    
    print(f'Loading "{os.path.basename(modelfilename)}"')
    checkpoint = torch.load(modelfilename)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print('Model Loaded!\n Created at train epoch: {}\n Train Loss: {:.4}'.format(checkpoint['epoch'], float(checkpoint['loss'])))
    test_normal_dataset = dataset(path_feature=args.path_feature_test, scanlabel=args.scanlabel_test, slicelabel=args.slicelabel_test,\
                                 classname='normal', feature_extracted_model=args.feature_extracted_model, max_slicelen=args.max_slicelen,\
                                 brainarea_ratio=args.brainarea_ratio)
    test_anomaly_dataset = dataset(path_feature=args.path_feature_test, scanlabel=args.scanlabel_test, slicelabel=args.slicelabel_test,\
                                classname='anomaly', feature_extracted_model=args.feature_extracted_model, max_slicelen=args.max_slicelen,\
                                brainarea_ratio=args.brainarea_ratio)   
    test_normal_loader = DataLoader(dataset=test_normal_dataset, batch_size=1, pin_memory=True, num_workers=4, shuffle=False)
    test_anomaly_loader = DataLoader(dataset=test_anomaly_dataset, batch_size=1, pin_memory=True, num_workers=4, shuffle=False)

    save_path = os.path.join(args.saveroot, args.model_name, args.feature_extracted_model, 'k_{}_Lambda_{}'.format(args.k, args.Lambda))
    if args.slicelabel_test is not None:
        test(test_normal_loader=test_normal_loader, test_anomaly_loader=test_anomaly_loader, model=model, device=device, save_path=save_path, args=args)
        eval_ichsubtype(test_normal_loader=test_normal_loader, test_anomaly_loader=test_anomaly_loader, model=model, device=device, save_path=save_path, args=args)
    else:
        test_no_slicelabel(test_normal_loader=test_normal_loader, test_anomaly_loader=test_anomaly_loader, model=model, device=device, save_path=save_path, args=args)