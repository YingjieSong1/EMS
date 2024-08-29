import numpy as np  
from sklearn import metrics
import codecs
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--result_root', type=str,default='./result')
args = parser.parse_args()

set_names=['Set_0','Set_1','Set_2','Set_3']
output_result=f'{args.result_root}/val.txt'
output_txt = codecs.open( output_result, 'w', 'utf-8')
set_names.sort()

set_name_list,acc_list,sentivity_list,specificity_list,auc_list,precision_list,F1_score_list=[],[],[],[],[],[],[]
for set_name in tqdm(set_names):
    scp_th = f'{args.result_root}/{set_name}/threshold.txt'
    with codecs.open(scp_th, 'r', 'utf-8') as scp_th:
        for line in scp_th:
            line = line.strip()
            threshold=float(line)

    scp = f'{args.result_root}/{set_name}/prob_val.txt'
    y_true_list,y_pred_list,y_score_list =[],[],[]
    with codecs.open(scp, 'r', 'utf-8') as scp:
        for line in scp:
            line = line.strip()
            y_true=int(line.split(',')[0])
            y_true=0 if y_true<200 else 1
            y_true_list.append(y_true)
            y_pred=float(line.split(',')[1])
            y_score_list.append(y_pred)
            y_pred=0 if y_pred<threshold else 1
            y_pred_list.append(y_pred)

    acc=metrics.accuracy_score(y_true=np.array(y_true_list) , y_pred=np.array(y_pred_list))  
    TN, FP, FN, TP=metrics.confusion_matrix(y_true=np.array(y_true_list), y_pred=np.array(y_pred_list)).ravel()  
    sentivity = TP / (TP+FN)
    specificity = TN / (TN+FP)
    precision=TP/(TP+FP)
    recall=sentivity
    F1_score=(2*precision*recall)/(recall+precision)
    auc=metrics.roc_auc_score(y_true=np.array(y_true_list), y_score=np.array(y_score_list))

    output_txt.write(f'{set_name}, acc:{acc}, sentivity:{sentivity}, specificity:{specificity}, auc:{auc}, precision:{precision}, F1_score:{F1_score}')
    output_txt.write('\n')

    set_name_list.append(set_name)
    acc_list.append(acc)
    sentivity_list.append(sentivity)
    specificity_list.append(specificity)
    auc_list.append(auc)
    precision_list.append(precision)
    F1_score_list.append(F1_score)

output_txt.write(f'average, acc:{np.average(acc_list)}, sentivity:{np.average(sentivity_list)}, specificity:{np.average(specificity_list)}, auc:{np.average(auc_list)}, precision:{np.average(precision_list)}, F1_score:{np.average(F1_score_list)}')
output_txt.write('\n')

index=np.argmax(auc_list)
output_txt.write(f'best:{set_name_list[index]}')

