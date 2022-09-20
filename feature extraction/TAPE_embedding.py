import torch
from tape import  ProteinBertModel,TAPETokenizer
import warnings

from sklearn.decomposition import PCA#在sklearn中调用PCA机器学习算法
from tqdm import tqdm

warnings.filterwarnings('ignore')
# from tape import TRRosetta
import numpy as np
model = ProteinBertModel.from_pretrained('bert-base')
tokenizer = TAPETokenizer(vocab='iupac')
labels=[]
sequence=[]
vec_lst=[]
np.set_printoptions(threshold=np.inf)
f = open("./IPVP_Datasets/iPVP data.txt", 'r', encoding="utf-8")
f1 = open("./IPVP_Datasets/result.txt",'w', encoding="utf-8")
lines = f.readlines()
for line in lines:
    sequence.append(line.split(' ')[0].strip())
    labels.append(line.split(' ')[1])
f.close()
for i in tqdm(sequence, desc='bert-base'):
    token_ids = torch.tensor([tokenizer.encode(i)])
    output = model(token_ids)
    sequence_output = output[0]
    b=sequence_output.sum(axis=0).mean(axis=0).reshape(1,768)
    #pca.fit(b.detach().numpy()) # 对基础数据集进行相关的计算，求取相应的主成分
    f1.writelines(str(b.tolist()).replace('[','').replace(']','').replace(',',' ')+'\n')
f1.close()
f3 = open("./IPVP_Datasets/result.txt", 'r', encoding="utf-8")
f4 = open("./IPVP_Datasets/ProSE/iPVP data_TAPE.txt", 'w', encoding="utf-8")
lines = f3.readlines()
for i,line in enumerate(lines):
    f4.writelines(line.strip()+" "+labels[i])
f4.close()
# import csv
