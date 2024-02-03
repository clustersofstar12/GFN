import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn as nn
import os, shutil, json
import argparse
from tools.Trainer import ModelNetTrainer
from tools.ImgDataset import MultiviewImgDataset, SingleImgDataset
from model.view_gfn import view_GFN, SVCNN
def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False
# os.environ['CUDA_VISIBLE_DEVICES']='2'
parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, help="Name of the experiment", default="view-gfn")
parser.add_argument("-bs", "--batchSize", type=int, help="Batch size for the second stage", default=20)# it will be *12 images in each batch for mvcnn
parser.add_argument("-num_models", type=int, help="number of models per class", default=0)
parser.add_argument("-lr", type=float, help="learning rate", default=0.001)
parser.add_argument("-weight_decay", type=float, help="weight decay", default=0.01)
parser.add_argument("-no_pretraining", dest='no_pretraining', action='store_true', default=True)
parser.add_argument("-cnn_name", "--cnn_name", type=str, help="cnn model name", default="resnet18")
parser.add_argument("-num_views", type=int, help="number of views", default=12)
parser.add_argument("-train_path", type=str, default="/home/lhy/rgbd/*/train")
#parser.add_argument("-train_path", type=str, default="/home/student/lhy/modelnet40_20/*/train")
parser.add_argument("-val_path", type=str, default="/home/lhy/rgbd/*/test")
#parser.add_argument("-val_path", type=str, default="/home/student/lhy/modelnet40_20/*/test")
parser.set_defaults(train=False)

def create_folder(log_dir):
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    else:
        print('WARNING: summary folder already exists!! It will be overwritten!!')
        shutil.rmtree(log_dir)
        os.mkdir(log_dir)

if __name__ == '__main__':
    seed_torch()
    args = parser.parse_args()
    pretraining = args.no_pretraining
    log_dir = args.name
    create_folder(args.name)
    config_f = open(os.path.join(log_dir, 'config.json'), 'w')
    json.dump(vars(args), config_f)
    config_f.close()
    # STAGE 1
    log_dir = args.name+'_stage_1'
    create_folder(log_dir)
    cnet = SVCNN(args.name, nclasses=51, pretraining=pretraining, cnn_name=args.cnn_name)
    n_models_train = args.num_models * args.num_views
    optimizer = optim.SGD(cnet.parameters(), lr=1e-2, weight_decay=0.01, momentum=0.9)
    train_dataset = SingleImgDataset(args.train_path, scale_aug=False, rot_aug=False, num_models=n_models_train, num_views=args.num_views)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=400, shuffle=True, num_workers=4)
    val_dataset = SingleImgDataset(args.val_path, scale_aug=False, rot_aug=False, test_mode=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=400, shuffle=False, num_workers=4)
    print('num_train_files: '+str(len(train_dataset.filepaths)))
    print('num_val_files: '+str(len(val_dataset.filepaths)))
    trainer = ModelNetTrainer(cnet, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'svcnn', log_dir, num_views=1)
    trainer.train(10)

    # # # STAGE 2CUDA_VISIBLE_DEVICES=2 python train.py
    log_dir = args.name+'_stage_2'
    create_folder(log_dir)
    cnet_2 = view_GFN(args.name, cnet, nclasses=51, cnn_name=args.cnn_name, num_views=args.num_views)
    optimizer = optim.SGD(cnet_2.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    train_dataset = MultiviewImgDataset(args.train_path, scale_aug=False, rot_aug=False, num_models=n_models_train, num_views=args.num_views,test_mode=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=False, num_workers=4)# shuffle needs to be false! it's done within the trainer
    val_dataset = MultiviewImgDataset(args.val_path, scale_aug=False, rot_aug=False, num_views=args.num_views,test_mode=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize, shuffle=False, num_workers=4)
    print('num_train_files: '+str(len(train_dataset.filepaths)))
    print('num_val_files: '+str(len(val_dataset.filepaths)))
    trainer = ModelNetTrainer(cnet_2, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'view-gfn', log_dir, num_views=args.num_views)
    #use trained_view_GFN
    #cnet_2.load_state_dict(torch.load('trained.pth'))
    #trainer.update_validation_accuracy(1)
    trainer.train(200)
    #torch.save(cnet_2.state_dict(), "trained.pth")