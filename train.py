import argparse
import os,cv2
import torch
import time
import numpy as np
from loss import *
from utils import *
from sklearn import metrics


def train(model, optimizer, loader, epoch, device, args):
    model.train()
    global tic_train_begin 
    
    total_loss = 0.0
    cur_loss = 0.0

    for idx, (gt,fix_feature) in enumerate(tqdm(loader)):
        global global_steps
        global_steps=global_steps+1

        gt = gt.to(device)
        gt = torch.unsqueeze(gt, dim=-1)
        fix_feature = fix_feature.to(device)
        
        optimizer.zero_grad()

        pred= model(fix_feature,alpha=min(1,global_steps/50))
        assert pred.size() == gt.size()
        loss= loss_func(pred, gt)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        cur_loss += loss.item()

        if idx%args.log_interval==0:
            tqdm.write('[{:2d}, {},{}] bce_loss : {:.5f}, time:{:.3f} minutes'.\
                  format(epoch, idx,global_steps, cur_loss/args.log_interval, (time.time()-tic_train_begin)/60))
            cur_loss = 0.0

    tqdm.write('[{:2d}, train] bce_loss : {:.5f}'.format(epoch, total_loss/len(loader)))
    return total_loss/len(loader)

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    pred_log=torch.empty(len(loader),1)
    gt_log=torch.empty(len(loader),1)
    global global_steps
    
    for idx, (gt,fix_feature) in enumerate(tqdm(loader)):
        gt = gt.to(device)
        gt = torch.unsqueeze(gt, dim=-1)
        fix_feature = fix_feature.to(device)
        pred= model(fix_feature,alpha=min(1,global_steps/50))
        assert pred.size() == gt.size()
        pred_log[idx,:]=pred
        gt_log[idx,:]=gt

    gt_log=np.array(gt_log.detach().cpu().numpy()[:,0])
    pred_log=np.array(pred_log.detach().cpu().numpy()[:,0])
    auc=metrics.roc_auc_score(y_true=gt_log, y_score=pred_log)
    pred_log_tmp=np.expand_dims(pred_log*255,axis=-1).astype(np.uint8)
    threshold_cv, _ = cv2.threshold(pred_log_tmp.copy(), 0, 255,  cv2.THRESH_OTSU)
    threshold_cv=float(threshold_cv)/255
    TN, FP, FN, TP=metrics.confusion_matrix(y_true=gt_log.astype(np.uint8), y_pred=(pred_log>=threshold_cv).astype(np.uint8)).ravel()
    acc=(TP+TN)/ (TP+FN+FP+TN+1e-8)
    sen = TP / (TP+FN+1e-8)
    spe = TN / (TN+FP+1e-8)
    precision=TP/(TP+FP+1e-8)
    F1_score=(2*precision*sen)/(sen+precision)
    
    acc_str='[{:2d}, val] auc:{:.5f}, acc:{:.5f}, sen:{:.5f}, spe:{:.5f}, pre:{:.5f}, F1_score:{:.5f}'.\
          format(epoch,auc, acc, sen, spe,precision,F1_score)
    tqdm.write(acc_str)
    
    return auc+acc,acc_str


##############################################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    
    parser.add_argument('--no_epochs',default=50, type=int)
    parser.add_argument('--log_interval',default=3, type=int)
    parser.add_argument('--batch_size',default=8, type=int)
    parser.add_argument('--no_workers',default=4, type=int)
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--module', type=str,default='MSNet')
    parser.add_argument('--feature_dict',default='feature_dict_RINet.npy', type=str)
    parser.add_argument('--fix_number',default=14, type=int)
    parser.add_argument('--fix_embedding',default=1056, type=int)
    parser.add_argument('--result_root',default=f"./result/")
    parser.add_argument('--set_name', type=str,default='Set_0')

    args,unknown = parser.parse_known_args()
    train_loader,val_loader= dataset_process(args)  
    model,device=create_model(args)
    optimizer,scheduler=create_optimizer_scheduler(model,args)

    global global_steps
    global_steps=-1
    best_model_path=os.path.join(args.result_root,args.set_name,"best_model.pt")
    tic_train_begin = time.time()
    best_acc=0
    for epoch in range(0, args.no_epochs):
        torch.cuda.empty_cache()
        if epoch != 0:
                model_dict = model.state_dict()
                pretrained_dict = torch.load(best_model_path)
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)

        train_loss = train(model, optimizer, train_loader, epoch, device, args)
        
        with torch.no_grad():
            acc,acc_str = validate(model, val_loader, device)

            if best_acc < acc:
                best_acc = acc
                print('[{:2d},  save, {}]'.format(epoch, best_model_path))
                with open(os.path.join(args.result_root,args.set_name,"best_model.txt"),"a") as f:
                        f.write(f'{acc_str}\n')
                torch.save(model.state_dict(), best_model_path)

        scheduler.step(epoch)