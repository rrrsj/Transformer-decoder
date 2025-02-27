import pandas as pd
from torch.utils import data
import torch
import re
from torch.utils.data import DataLoader
import math
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm,trange
from datetime import datetime
import copy
import json
import numpy as np
import random
import pyarrow as pa

class MyData(data.Dataset):

    def __init__(self):
        self.input=[]
        self.path=['./dataset/OmniData___zhihu-kol/default-c94fa9fb124e14f3/0.0.0/master/zhihu-kol-train-00000-of-00005.arrow',
        './dataset/OmniData___zhihu-kol/default-c94fa9fb124e14f3/0.0.0/master/zhihu-kol-train-00001-of-00005.arrow' ,
        './dataset/OmniData___zhihu-kol/default-c94fa9fb124e14f3/0.0.0/master/zhihu-kol-train-00002-of-00005.arrow' ,
        './dataset/OmniData___zhihu-kol/default-c94fa9fb124e14f3/0.0.0/master/zhihu-kol-train-00003-of-00005.arrow' ,
        './dataset/OmniData___zhihu-kol/default-c94fa9fb124e14f3/0.0.0/master/zhihu-kol-train-00004-of-00005.arrow' ]
        self.get_data()

    def get_data(self):
        for now_path in self.path:
            df=self.read_arrow_to_df(now_path)
            for j in range(len(df['INSTRUCTION'])):
                self.input.append(df['INSTRUCTION'][j]+df['RESPONSE'][j])
    
    
    def __getitem__(self,index):

        return self.input[index]+"<|endoftext|>"
    
    def __len__(self):
        return len(self.input)


    def read_arrow_to_df(self,path):
        with open(path, "rb") as f:
            reader = pa.ipc.RecordBatchStreamReader(f)
            df = reader.read_pandas()
        return df