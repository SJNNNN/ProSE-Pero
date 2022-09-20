import time

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, recall_score, matthews_corrcoef
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader,Dataset,TensorDataset
from sklearn.metrics import f1_score, accuracy_score, precision_score
import torch
import warnings
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import KFold
warnings.filterwarnings('ignore')
#FastText
class Net(nn.Module):
    def __init__(self, w2v_dim, classes, hidden_size):
        super(Net, self).__init__()
        #创建embedding
        # self.embed = nn.Embedding(len(vocab), w2v_dim)  #embedding初始化，需要两个参数，词典大小、词向量维度大小
        # self.embed.weight.requires_grad = True #需要计算梯度，即embedding层需要被训练
        self.fc = nn.Sequential(              #序列函数
            nn.Linear(w2v_dim, hidden_size),  #这里的意思是先经过一个线性转换层
            nn.BatchNorm1d(hidden_size),      #再进入一个BatchNorm1d
            nn.ReLU(inplace=True),            #再经过Relu激活函数
            nn.Linear(hidden_size, classes).float()#最后再经过一个线性变换
        )
    def forward(self, x):
        # x = self.embed(x)                     #先将词id转换为对应的词向量
        out = self.fc(torch.mean(x.float(), dim=1))  #这使用torch.mean()将向量进行平均
        return out
#CNNBiLSTM
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=10, kernel_size=3, stride=2).double()
        self.max_pool1 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.lstm = nn.LSTM(394, 100, num_layers=1, dropout=0.5,
                       bidirectional=True).double()
        self.liner1 = nn.Linear(200,1).double()
        self.liner2 = nn.Linear(10, 2).double()
    def forward(self,x):
        output = self.conv1(x)
        output = self.max_pool1(output)
        hidden_cell = (torch.zeros([2, 10, 100], dtype=torch.double), torch.zeros([2, 10, 100], dtype=torch.double))
            # x.view(-1,40 * 14)
        lstm_out, (h_n, h_c) =self.lstm(output, hidden_cell)
        output = self.liner1(lstm_out)
        output = output.permute(0, 2, 1)
        output = self.liner2(output)
        output = output.squeeze(1)
        output = F.softmax(output)
        return output
#CNNBilstm+Attention
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=10, kernel_size=3, stride=2).double()
        self.max_pool1 = nn.MaxPool1d(kernel_size=3, stride=2)
        # self.conv2 = nn.Conv1d(in_channels=10, out_channels=10, kernel_size=3, stride=2).double()
        # self.max_pool2 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.lstm = nn.LSTM(5, 100, num_layers=1, dropout=0.5,
                       bidirectional=True).double()
        self.liner1 = nn.Linear(200,2).double()
        # self.liner2 = nn.Linear(10, 2).double()
    def forward(self,X):

            output = self.conv1(X)
            output = self.max_pool1(output)
            # output = self.conv2(output)
             # output = self.max_pool2(output)
            hidden_cell = (torch.zeros([2, 10, 100], dtype=torch.double), torch.zeros([2, 10, 100], dtype=torch.double))
            # x.view(-1,40 * 14)
            lstm_out, (h_n, h_c) =self.lstm(output, hidden_cell)
            # output = self.liner1(lstm_out)
            x = lstm_out  # .permute(1, 0, 2).double()
            # print(x.shape)
            hidden_layer_size = 100
            # x形状是(batch_size, seq_len, 2 * num_hiddens)
            w_omega = nn.Parameter(torch.DoubleTensor(
                X.shape[0], hidden_layer_size*2, hidden_layer_size*2))
            u_omega = nn.Parameter(torch.DoubleTensor(X.shape[0], hidden_layer_size*2, 1))
            nn.init.uniform_(w_omega, -0.1, 0.1)
            nn.init.uniform_(u_omega, -0.1, 0.1)
            # print(w_omega.shape)
            # Attention过程
            u = torch.tanh(torch.matmul(x, w_omega))
            # u形状是(batch_size, seq_len, 2 * num_hiddens)
            att = torch.matmul(u, u_omega)
            # att形状是(batch_size, seq_len, 1)
            att_score = F.softmax(att, dim=1)
            # att_score形状仍为(batch_size, seq_len, 1)
            scored_x = x * att_score
            # scored_x形状是(batch_size, seq_len, 2 * num_hiddens)
            # Attention过程结束
            feat = torch.sum(scored_x, dim=1).double()
            output= self.liner1(feat)
            output = F.softmax(output)
            return output
#TextCNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=10, kernel_size=1, stride=1).double()
        self.max_pool1 = nn.MaxPool1d(kernel_size=1, stride=2)
        self.conv2 = nn.Conv1d(10, 10, 1, 1).double()
        self.max_pool2 = nn.MaxPool1d(kernel_size=1, stride=1)
        self.liner1 = nn.Linear(13, 1).double()
        self.liner2 = nn.Linear(10, 2).double()
    def forward(self,x):
            x = self.conv1(x)
            x = self.max_pool1(x)
            x = self.conv2(x)
            x = self.max_pool2(x)
            # x.view(-1,40 * 14)
            x = self.liner1(x)
            x = x.squeeze(2)
            x = self.liner2(x)
            x = F.softmax(x)
            return x
path= "../../IPVP_Datasets/fusion/"
dataset="ProSETAPEFusion_ANOVA.txt"
idx_features_labels = np.genfromtxt("{}{}".format(path, dataset),
                               dtype=np.dtype(str))

X_train=idx_features_labels[0:400, 0:-1]
Y_train=idx_features_labels[0:400, -1]
X_test=idx_features_labels[400:548,0:-1]
Y_test=idx_features_labels[400:548,-1]
x_train=[]
x_test=[]
for i in range(len(X_train)):
       x_train1 = []
       m=X_train[i]
       for j in range(len(m)):
              x_train1.append(float(m[j]))
       x_train.append(x_train1)
for i in range(len(X_test)):
       x_test1 = []
       m=X_test[i]
       for j in range(len(m)):
              x_test1.append(float(m[j]))
       x_test.append(x_test1)
y_train_smo=[]
y_test_smo=[]
for i in Y_train:
    y_train_smo.append(float(i))
for i in Y_test:
    y_test_smo.append(float(i))

batch_size = 8
epoch = 10  # 迭代次数
w2v_dim = 792# 词向量维度
lr = 0.001
hidden_size = 128
classes = 2

def perf_measure(y_true, y_pred):
    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
           TP += 1
        if y_true[i] == 0 and y_pred[i] == 1:
           FP += 1
        if y_true[i] == 0 and y_pred[i] == 0:
           TN += 1
        if y_true[i] == 1 and y_pred[i] == 0:
           FN += 1
    return TP, FP, TN, FN
ACC1 = []
Precision_Scores1= []
F1_Scores1= []
Recall_Scores1= []
SP1 = []
Mccc1= []
l1=[]
p1=[]
KF = KFold(n_splits=5,shuffle=True,random_state=100)
for train_index,test_index in KF.split(x_train):
    # print(train_index)
    train_data = TensorDataset(torch.from_numpy(np.array(x_train)[train_index]), torch.from_numpy(np.array(y_train_smo)[train_index]))
    valid_data = TensorDataset(torch.from_numpy(np.array(x_train)[test_index]), torch.from_numpy(np.array(y_train_smo)[test_index]))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=10,drop_last=True)
    test_loader = DataLoader(valid_data, shuffle=True, batch_size=10,drop_last=True)
    ACC = []
    Precision_Scores = []
    F1_Scores = []
    Recall_Scores = []
    SP = []
    Mccc = []
# if __name__ == '__main__':
#     train_loader = DataLoader(train_data, shuffle=True, batch_size=8)
#     test_loader = DataLoader(valid_data, shuffle=True, batch_size=8)

    # 建模三件套：loss，优化，epochs
    # model = Net()  # 模型
    # model = ResNet(in_channels=1)
    model  = Net(w2v_dim=w2v_dim, classes=classes, hidden_size=hidden_size)
    loss_function = nn.CrossEntropyLoss()  # loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 优化器
    epochs = 101
    # 开始训练
    model.train()
    for i in range(epochs):
        acc1=[]
        precision_scores = []
        f1_scores=[]
        recall_scores=[]
        sp1=[]
        MCC1=[]
        for seq, labels in train_loader:
            optimizer.zero_grad()
            #seq.unsqueeze(1).double()
            y_pred = model(seq.unsqueeze(1).double())#.squeeze()
            # 压缩维度：得到输出，并将维度为1的去除
            correct = torch.eq(torch.max(y_pred, dim=1)[1], labels).float()
            acc = correct.sum() / len(correct)
            acc1.append(acc)
            precision_scores.append(precision_score(labels, torch.max(y_pred, dim=1)[1]))
            f1_scores.append(f1_score(labels, torch.max(y_pred, dim=1)[1]))
            recall_scores.append(recall_score(labels, torch.max(y_pred, dim=1)[1]))
            single_loss = loss_function(y_pred, labels.long())
            TP, FP, TN, FN = perf_measure(labels, torch.max(y_pred, dim=1)[1])
            if((TN+FP)!=0):
              Sp = TN / (TN + FP)
            else:
              Sp=0
            sp1.append(Sp)
            MCC = matthews_corrcoef(labels, torch.max(y_pred, dim=1)[1])
            MCC1.append(MCC)
            # 若想要获得类别，二分类问题使用四舍五入的方法即可：print(torch.round(y_pred))
            single_loss.backward()
            optimizer.step()
        # print("Train Step:", i, " acc:{:.6f}, pre:{:.4f},f1score:{:.4f},Sn:{:.4f},Sp:{:.4f},MCC:{:.4f} " .format(np.array(acc1).mean(), np.array(precision_scores).mean(), np.array(f1_scores).mean()
        #   , np.array(recall_scores).mean(), np.array(sp1).mean(), np.array(MCC1).mean()))
    # 开始验证
    model.eval()
    acc2 = []
    precision_scores = []
    f1_scores = []
    recall_scores = []
    sp1 = []
    MCC1 = []
    # f = open("test-cache/CNNBiLSTM_10_result.txt", 'w', encoding="utf-8")
    # f = open("PP/112.txt", 'w', encoding="utf-8")
    for i in range(epochs):
        for seq, labels in test_loader:  # 这里偷个懒，就用训练数据验证哈！

            y_pred = model(seq.unsqueeze(1).double())#.squeeze()  # 压缩维度：得到输出，并将维度为1的去除
            correct = torch.eq(torch.max(y_pred, dim=1)[1], labels).float()
            acc = correct.sum() / len(correct)
            # p = torch.max(y_pred, dim=1)[1].numpy().tolist()
            p = y_pred.detach().numpy().tolist()
            l = labels.numpy().tolist()
            for j in range(len(p)):
                # f.writelines(str(p[j][1]) + " " + str(int(l[j])) + "\n")
                p1.append(p[j])
                l1.append(int(l[j]))
            acc2.append(acc)
            precision_scores.append(precision_score(labels, torch.max(y_pred, dim=1)[1]))
            f1_scores.append(f1_score(labels, torch.max(y_pred, dim=1)[1]))
            recall_scores.append(recall_score(labels, torch.max(y_pred, dim=1)[1]))
            single_loss = loss_function(y_pred, labels.long())
            TP, FP, TN, FN = perf_measure(labels, torch.max(y_pred, dim=1)[1])
            if ((TN + FP) != 0):
                Sp = TN / (TN + FP)
            else:
                Sp = 0
            sp1.append(Sp)
            MCC = matthews_corrcoef(labels, torch.max(y_pred, dim=1)[1])
            MCC1.append(MCC)
        ACC.append(np.mean(acc2))
        F1_Scores.append(np.mean(f1_scores))
        Precision_Scores.append(np.mean(precision_scores))
        Recall_Scores.append(np.mean(recall_scores))
        SP.append(np.mean(sp1))
        Mccc.append(np.mean(MCC1))
        # print("Test Step:", i, " acc:{:.6f}, pre:{:.4f},f1score:{:.4f},Sn:{:.4f},Sp:{:.4f},MCC:{:.4f} ".format(np.array(acc2).mean(), np.array(precision_scores).mean(), np.array(f1_scores).mean()
        #   , np.array(recall_scores).mean(), np.array(sp1).mean(), np.array(MCC1).mean()))
    # f.close()
    print(" acc:{:.6f}, pre:{:.4f},f1score:{:.4f},Sn:{:.4f},Sp:{:.4f},MCC:{:.4f} ".format(np.mean(ACC), np.mean(Precision_Scores), np.mean(F1_Scores)
          , np.mean(Recall_Scores), np.mean(SP), np.mean(Mccc)))
    ACC1.append(np.mean(ACC))
    Precision_Scores1.append(np.mean(Precision_Scores))
    F1_Scores1.append(np.mean(F1_Scores))
    Recall_Scores1.append(np.mean(Recall_Scores))
    SP1.append(np.mean(SP))
    Mccc1.append(np.mean(Mccc))
print(" acc:{:.6f}, pre:{:.4f},f1score:{:.4f},Sn:{:.4f},Sp:{:.4f},MCC:{:.4f} ".format(np.mean(ACC1), np.mean(Precision_Scores1), np.mean(F1_Scores1)
          , np.mean(Recall_Scores1), np.mean(SP1), np.mean(Mccc1)))
import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc  ###计算roc和auc

    # p = []
    # l = []
    # f = open("test-cache/0.txt", 'r', encoding="utf-8")
    # Text_lines = f.readlines()
    # for i in Text_lines:
    #     p.append(i.split('\t')[0])
    #     l.append(int(i.strip().split('\t')[1]))
    # # print([np.argmax(y) for y in p])
    # for i in p:
    #     print(i)
    #     print(type(i))
    #     print(np.argmax(i))
    # print([y[np.argmax(y)] for y in p1])
f= open("../../IPVP_Datasets/fasttext_5fold_result.txt", 'w', encoding="utf-8")
for j in range(len(p1)):
    f.writelines(str(p1[j][1]) + " " + str(int(l1[j])) + "\n")
f.close()
fpr, tpr, threshold = roc_curve(l1, [y[1] for y in p1])  ###计算真正率和假正率
roc_auc = auc(fpr, tpr)  ###计算auc的值
lw = 2
plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.6f)' % roc_auc)   ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


