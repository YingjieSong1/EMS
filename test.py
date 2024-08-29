import argparse
import os
import torch
from dataloader import TestLoader
from utils import test_model
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str)
parser.add_argument('--module',default="MSNet", type=str)
parser.add_argument('--feature_dict',default="feature_dict_RINet.npy", type=str)
parser.add_argument('--fix_number',default=14, type=int)
parser.add_argument('--fix_embedding',default=1056, type=int)
parser.add_argument('--no_workers',default=4, type=int)
parser.add_argument('--result_root', type=str,default='./result')
parser.add_argument('--set_name', type=str,default='Set_1')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.module == "MSNet":
    from MSNet import Model     
    model = Model(in_c=args.fix_embedding)
else:
    raise AssertionError('Please check args.module!')
weights=torch.load(os.path.join(args.result_root,args.set_name,'best_model.pt'))
weights_dict = {}
for k, v in weights.items():
    new_k = k.replace('module.', '') if 'module' in k else k
    weights_dict[new_k] = v

model.load_state_dict(weights_dict)
model = model.to(device)
   
subject_ids=os.listdir(os.path.join(args.dataset_dir,'Test/Fixations'))
subject_ids=[x.split('.')[0] for x in subject_ids]
feature_dict = np.load(f'{args.feature_dict}', allow_pickle=True).item()

test_dataset = TestLoader(set_ids=subject_ids,feature_dict=feature_dict,args=args)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.no_workers)
test_model(model, test_loader, device, args)

    







