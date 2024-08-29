import os,glob
import pandas as pd
from tqdm import tqdm
import numpy as np
import codecs
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str)
args = parser.parse_args()

output_path=f'{args.dataset_dir}/TXT'
input_path_val=f'{args.dataset_dir}/Train_Valid/Fixations/*'
input_path_test=f'{args.dataset_dir}/Test/Fixations/*'
os.makedirs(output_path,exist_ok=True)

for f_path in tqdm(glob.glob(input_path_val)+glob.glob(input_path_test)):
    subject_index=os.path.basename(f_path).split('.')[0]
    df=pd.read_excel(io=f_path)
    IMAGE_list=df['IMAGE'].values.tolist()
    Fix_index_list=df['FIX_INDEX'].values
    Fix_duration_list=df['FIX_DURATION'].values
    FIX_X_list=df['FIX_X'].values
    FIX_Y_list=df['FIX_Y'].values

    Images_in_xlsx=np.unique(IMAGE_list)
    Images = []
    for home, dirs, files in os.walk(f'{args.dataset_dir}/Images'):
        for filename in files:
            Images.append(filename)


    for image in Images:
        image_name=image.split('.')[0]
        folder_name=subject_index+'_'+image_name

        if image not in Images_in_xlsx:
            output_file = codecs.open( os.path.join(output_path,f'{folder_name}.txt') , 'w', 'utf-8')
        else:
            index=[i for i,x in enumerate(IMAGE_list) if x==image]
            FIX_index=Fix_index_list[index]
            FIX_X=np.floor(FIX_X_list[index]).astype(np.int64)
            FIX_Y=np.floor(FIX_Y_list[index]).astype(np.int64)
            FIX_duration=Fix_duration_list[index]

            out_index_X= [i for i,x in enumerate(FIX_X) if x>=1024]
            out_index_Y= [i for i,x in enumerate(FIX_Y) if x>=768]
            out_index=list(np.unique(out_index_X+out_index_Y)) 

            if out_index!=[]:
                FIX_X=np.delete(FIX_X, out_index, axis=0)
                FIX_Y=np.delete(FIX_Y, out_index, axis=0)
                FIX_index=np.delete(FIX_index, out_index, axis=0)
                FIX_duration=np.delete(FIX_duration, out_index, axis=0)

            output_file = codecs.open(os.path.join(output_path,f'{folder_name}.txt') , 'w', 'utf-8')
            for i in range(len(FIX_index)):
                line=str(FIX_index[i])+','+str(FIX_X[i])+','+str(FIX_Y[i])+','+str(FIX_duration[i])
                output_file.write(line)
                output_file.write('\n')
        