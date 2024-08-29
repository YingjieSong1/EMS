from torch.utils.data import DataLoader
import numpy as np
import os
import pandas as pd


class BaseLoader(DataLoader):
    def __init__(self,set_ids=None,feature_dict=None,args=None):

        self.set_ids=set_ids
        self.feature_dict=feature_dict
        self.dataset_dir=args.dataset_dir
        self.image_list=feature_dict.keys()
        self.fix_number=args.fix_number
        self.channel=args.fix_embedding
        self.image_number=100
        self.image_size=(768,1024)

    def __getitem__(self, idx):
        suject_id = self.set_ids[idx]
        if int(suject_id)<200:
            label=0
        else:
            label=1
        
        fix_feature=(-1)*np.ones((self.image_number,self.fix_number,self.channel))
        for image_index,image_name in enumerate(self.image_list):
            image_feature=self.feature_dict[image_name]
            H_f,W_f,_=image_feature.shape
            H_i,W_i=self.image_size
            fix_path=os.path.join(self.dataset_dir,'TXT',suject_id+'_'+image_name+'.txt')
            if os.path.getsize (fix_path)!=0:
                fix_information = pd.read_csv(fix_path, header=None).to_numpy()
                x_list=np.floor(fix_information[:,1]/W_i*W_f).astype(np.int32)
                y_list=np.floor(fix_information[:,2]/H_i*H_f).astype(np.int32)
                if len(x_list)>self.fix_number:
                    x_list=x_list[:self.fix_number]
                    y_list=y_list[:self.fix_number]
                len_fix=len(x_list)
                fix_feature[image_index,:len_fix,:]=image_feature[y_list,x_list,:]
            
        # B H W C .. B  100 14 1056
        return  label,np.transpose(fix_feature, (2,0,1)).astype(np.float32)
        
        
    def __len__(self):
        return len(self.set_ids)    


class TestLoader(DataLoader):
    def __init__(self,set_ids=None,feature_dict=None,args=None,img_list_need=False):

        self.set_ids=set_ids
        self.feature_dict=feature_dict
        self.dataset_dir=args.dataset_dir
        self.image_list=feature_dict.keys()
        self.fix_number=args.fix_number
        self.channel=args.fix_embedding
        self.image_number=100
        self.image_size=(768,1024)
        self.img_list_need=img_list_need

    def __getitem__(self, idx):
        suject_id = self.set_ids[idx]
        
        fix_feature=(-1)*np.ones((self.image_number,self.fix_number,self.channel))
        for image_index,image_name in enumerate(self.image_list):
            image_feature=self.feature_dict[image_name]
            H_f,W_f,_=image_feature.shape
            H_i,W_i=self.image_size
            fix_path=os.path.join(self.dataset_dir,'TXT',suject_id+'_'+image_name+'.txt')
            if os.path.getsize (fix_path)!=0:
                fix_information = pd.read_csv(fix_path, header=None).to_numpy()
                x_list=np.floor(fix_information[:,1]/W_i*W_f).astype(np.int32)
                y_list=np.floor(fix_information[:,2]/H_i*H_f).astype(np.int32)
                if len(x_list)>self.fix_number:
                    x_list=x_list[:self.fix_number]
                    y_list=y_list[:self.fix_number]
                len_fix=len(x_list)
                fix_feature[image_index,:len_fix,:]=image_feature[y_list,x_list,:]
            
        # B H W C .. B  100 14 1056
        if self.img_list_need==True:
            return  suject_id,np.transpose(fix_feature, (2,0,1)).astype(np.float32),list(self.image_list)
        else:
            return  suject_id,np.transpose(fix_feature, (2,0,1)).astype(np.float32)
    def __len__(self):
        return len(self.set_ids) 

