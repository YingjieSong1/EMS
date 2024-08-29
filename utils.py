import os
import torch
from tqdm import tqdm
from dataloader import BaseLoader
from loss import *
import pandas as pd
import numpy as np
import codecs
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler


def create_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    print(f'module:{args.module} ')
    
    if args.module == "MSNet":
        from MSNet import Model     
        model = Model(in_c=args.fix_embedding)
    else:
        raise AssertionError('Please check args.module!')
    model.to(device)
    
    import shutil
    if os.path.exists(os.path.join(args.result_root,args.set_name)):
        shutil.rmtree(os.path.join(args.result_root,args.set_name))
    os.makedirs(os.path.join(args.result_root,args.set_name),exist_ok=True)
    
    return model,device

def create_optimizer_scheduler(model,args):
    optimizer = create_optimizer(args, model)
    lr_scheduler, _ = create_scheduler(args, optimizer)
    return optimizer,lr_scheduler


def dataset_process(args):

    feature_dict = np.load(f'{args.feature_dict}', allow_pickle=True).item()
    df=pd.read_excel(io=os.path.join(args.dataset_dir,'Train_Valid.xlsx') )
    valid_subject_list=df[args.set_name].values.tolist()
    valid_subject_list=[str(i).rjust(3,'0') for i in valid_subject_list]
    Set_list=['Set_0','Set_1','Set_2','Set_3']
    train_set_name=[i for i in Set_list if i != args.set_name]
    print(f'train_set:{train_set_name} val_set:{args.set_name}')
    train_subject_list=[]
    for i in train_set_name:
        train_subject_list=train_subject_list+df[i].values.tolist() 
    train_subject_list=[str(i).rjust(3,'0') for i in train_subject_list]

    train_dataset = BaseLoader(set_ids=train_subject_list,feature_dict=feature_dict,args=args)
    val_dataset =   BaseLoader(set_ids=valid_subject_list,feature_dict=feature_dict,args=args)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.no_workers,pin_memory = True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.no_workers, pin_memory = True)
    return train_loader,val_loader


def val_model(model, loader, device, args):
    with torch.no_grad():

        model.eval()
        lines=[]

        for (suject_id,fix_feature) in tqdm(loader):
            fix_feature = fix_feature.to(device)
            pred= model(fix_feature)
            lines.append(f'{suject_id[0]},{pred.detach().cpu().item()}')

    output_txt = codecs.open(os.path.join(args.result_root,args.set_name,"prob_val.txt"), 'w', 'utf-8')
    for line in lines:
        output_txt.write(line)
        output_txt.write('\n')

def test_model(model, loader, device, args):
    scp_th = os.path.join(args.result_root,args.set_name,"threshold.txt") 
    with codecs.open(scp_th, 'r', 'utf-8') as scp_th:
        for line in scp_th:
            line = line.strip()
            threshold=float(line)

    with torch.no_grad():
        model.eval()
        lines_prob=[]

        for (suject_id,fix_feature) in tqdm(loader):
            fix_feature = fix_feature.to(device)
            prob= model(fix_feature).detach().cpu().item()
            pred=0 if prob<threshold else 1
            lines_prob.append(f'{suject_id[0]},{prob},{pred}')

    output_txt = codecs.open(os.path.join(args.result_root,args.set_name,"prob_test.txt"), 'w', 'utf-8')
    for line_prob in lines_prob:
        output_txt.write(line_prob)
        output_txt.write('\n')

            



