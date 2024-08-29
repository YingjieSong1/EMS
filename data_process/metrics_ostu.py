import numpy as np  
import codecs,os
from tqdm import tqdm
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--result_root', type=str,default='./result')
args = parser.parse_args()

set_names=os.listdir(args.result_root)
set_names.sort()

for set_name in tqdm(set_names):
    if '.' not in set_name:
        scp = f'{args.result_root}/{set_name}/prob_val.txt'
        y_score_list =[]
        with codecs.open(scp, 'r', 'utf-8') as scp:
            for line in scp:
                line = line.strip()
                y_score=float(line.split(',')[1])
                y_score_list.append(y_score)

        y_score_list=np.array(y_score_list)
        y_score_tmp=np.expand_dims(y_score_list*255,axis=-1).astype(np.uint8)
        threshold_cv, _ = cv2.threshold(y_score_tmp.copy(), 0, 255,  cv2.THRESH_OTSU)
                
        output_result=f'{args.result_root}/{set_name}/threshold.txt'
        output_txt = codecs.open( output_result, 'w', 'utf-8')
        output_txt.write(f'{float(threshold_cv)/255}')


