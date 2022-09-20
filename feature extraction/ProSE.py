import numpy as np
import torch
from tqdm import tqdm

from prose import fasta
from prose.alphabets import Uniprot21
def embed_sequence(model, x, pool='avg', use_cuda=False):
    if len(x) == 0:
        n = model.embedding.proj.weight.size(1)
        z = np.zeros((1,n), dtype=np.float32)
        return z

    alphabet = Uniprot21()
    x = x.upper()
    # convert to alphabet index
    x = alphabet.encode(x)
    x = torch.from_numpy(x)
    if use_cuda:
        x = x.cuda()

    # embed the sequence
    with torch.no_grad():
        x = x.long().unsqueeze(0)
        z = model.transform(x)
        # pool if needed
        z = z.squeeze(0)
        if pool == 'sum':
            z = z.sum(0)
        elif pool == 'max':
            z,_ = z.max(0)
        elif pool == 'avg':
            z = z.mean(0)
        z = z.cpu().numpy()

    return z
from prose.models.multitask import ProSEMT
model = ProSEMT.load_pretrained()
from prose.models.lstm import SkipLSTM
model1 = SkipLSTM.load_pretrained()
print("prose_mt")
# with open('data/demo.fa', 'rb') as f:
#     for name, sequence in fasta.parse_stream(f):
#         pid = name.decode('utf-8')
#         # print(sequence)
#         z = embed_sequence(model, sequence, pool='sum', use_cuda=False)
#         print(z)
# for name, sequence in fasta.parse_stream(f):
#     pid = name.decode('utf-8')
#     print(sequence)
labels=[]
sequence=[]
vec_lst=[]
np.set_printoptions(threshold=np.inf)
f = open("./IPVP_Datasets/iPVP data.txt", 'r', encoding="utf-8")
f1 = open("./IPVP_Datasets/result.txt",'w', encoding="utf-8")
lines = f.readlines()
for line in lines:
    sequence.append(line.split(' ')[0].strip())
    labels.append(line.split(' ')[1].strip())
f.close()
for i in tqdm(sequence):
    j = bytes(i, encoding='utf-8')
    z = embed_sequence(model, j, pool='avg', use_cuda=False)
    b = np.reshape(z, (1, 6165))
    f1.writelines(str(b.tolist()).replace('[','').replace(']','').replace(',',' ')+'\n')
f1.close()
f3 = open("./IPVP_Datasets/result.txt", 'r', encoding="utf-8")
f4 = open("./IPVP_Datasets/ProSE/iPVP data_ProSE.txt", 'w', encoding="utf-8")
lines = f3.readlines()
for i,line in enumerate(lines):
    f4.writelines(line.strip()+" "+labels[i]+"\n")
f3.close()
f4.close()
print("prose_dlm")
# with open('data/demo.fa', 'rb') as f:
#     for name, sequence in fasta.parse_stream(f):
#         pid = name.decode('utf-8')
#         # print(sequence)
#         z = embed_sequence(model1, sequence, pool='sum', use_cuda=False)
#         print(z)
# labels=[]
# sequence=[]
# vec_lst=[]
# np.set_printoptions(threshold=np.inf)
# f = open("./test-cache/test.txt", 'r', encoding="utf-8")
# f1 = open("./test-cache/ProSE_test_result.txt",'w', encoding="utf-8")
# lines = f.readlines()
# for line in lines:
#     sequence.append(line.split('\t')[0].strip())
#     labels.append(line.split('\t')[2])
# f.close()
# for i in tqdm(sequence):
#     j = bytes(i, encoding='utf-8')
#     z = embed_sequence(model1, j, pool='avg', use_cuda=False)
#     b = np.reshape(z, (1, 6165))
#     f1.writelines(str(b.tolist()).replace('[','').replace(']','').replace(',',' ')+'\n')
# f1.close()
# f3 = open("./test-cache/ProSE_test_result.txt", 'r', encoding="utf-8")
# f4 = open("./test-cache/ProSE_test_result_end.txt",'w', encoding="utf-8")
# lines = f3.readlines()
# for i,line in enumerate(lines):
#     f4.writelines(line.strip()+" "+labels[i])
# f3.close()
# f4.close()