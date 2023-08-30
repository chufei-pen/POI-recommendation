import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

#GCN層
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

#GCN
class GCN(nn.Module):
    def __init__(self, dropout, size):
        super(GCN, self).__init__()

        self.gcn = nn.ModuleList()
        self.dropout = dropout
        self.leaky_relu = nn.LeakyReLU(0.2)
        #self.x = nn.Parameter(torch.FloatTensor(size, 64))

        channels = [64,64,64]
        for i in range(len(channels) - 1):
            gcn_layer = GraphConvolution(channels[i], channels[i + 1])
            self.gcn.append(gcn_layer)

    def forward(self, adj, x):
        for i in range(len(self.gcn) - 1):
            x = self.leaky_relu(self.gcn[i](x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn[-1](x, adj)
        return x

#GCN，輸入包含使用者數量*圖
class batchGCN(nn.Module):
    def __init__(self, dropout, size):
        super(batchGCN, self).__init__()

        self.gcn = nn.ModuleList()
        self.dropout = dropout
        self.leaky_relu = nn.LeakyReLU(0.2)
        #self.x = nn.Parameter(torch.FloatTensor(size, 64))

        channels = [64,64,64]
        for i in range(len(channels) - 1):
            gcn_layer = GraphConvolution(channels[i], channels[i + 1])
            self.gcn.append(gcn_layer)

    def forward(self, batch, bx):
        out=[]
        for adj in batch:
            x=bx
            for i in range(len(self.gcn) - 1):
                x = self.leaky_relu(self.gcn[i](x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gcn[-1](x, adj)
            out.append(x)
        out=torch.stack(out)
        return out
    
#普通的lstm
class CustomLSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz):
        super().__init__()
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.W = nn.Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_sz * 4))
        self.init_weights()
                
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
         
    def forward(self, x, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device), 
                        torch.zeros(bs, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states
         
        HS = self.hidden_size
        
        #print(self.W.shape)
        #print(x.size)
        
        for t in range(seq_sz):
            x_t = x[:, t, :]
            # batch the computations into a single matrix multiplication
            
            gates = x_t @ self.W + h_t @ self.U + self.bias
            
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]), # input
                torch.sigmoid(gates[:, HS:HS*2]), # forget
                torch.tanh(gates[:, HS*2:HS*3]),
                torch.sigmoid(gates[:, HS*3:]), # output
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)
    

class DTTCG(nn.Module):

    def __init__(self, poiPad, catPad, userNum):
        super(DTTCG, self).__init__()

        self.feature_dim = self.hidden_dim = 64
        self.cat_dim = catPad+1
        self.user_dim = userNum+1
        self.poi_dim = poiPad+1
        self.dropout=0.2

        self.lstmC=CustomLSTM(self.feature_dim, self.hidden_dim)
        self.lstmP=CustomLSTM(self.feature_dim, self.hidden_dim)
        self.lstmT=CustomLSTM(self.feature_dim, self.hidden_dim)

        self.catkey = nn.Linear(self.hidden_dim , self.hidden_dim)
        self.catquery = nn.Linear(self.hidden_dim , self.hidden_dim)
        self.timekey = nn.Linear(self.hidden_dim , self.hidden_dim)
        self.timequery = nn.Linear(self.hidden_dim , self.hidden_dim)

        self.fccrep = nn.Linear(self.hidden_dim , self.hidden_dim)
        self.fctrep = nn.Linear(self.hidden_dim , self.hidden_dim)
        self.fcPc = nn.Linear(self.hidden_dim , self.hidden_dim)
        self.fcPt = nn.Linear(self.hidden_dim , self.hidden_dim)

        self.fcfuse = nn.Linear(self.hidden_dim*2 , self.hidden_dim)
        self.fcfuse2 = nn.Linear(self.hidden_dim , self.cat_dim)
        self.fcIp1 = nn.Linear(self.hidden_dim*3 , self.hidden_dim)
        self.fcIp2 = nn.Linear(self.hidden_dim , self.poi_dim)

        self.fcIp1 = nn.Linear(self.hidden_dim*4 , self.hidden_dim*4)
        self.fcIp2 = nn.Linear(self.hidden_dim*4 , self.poi_dim)

        self.fcIpc = nn.Linear(self.hidden_dim , self.hidden_dim)
        self.fcIpt = nn.Linear(self.hidden_dim , self.hidden_dim)
        self.fcIpct = nn.Linear(self.hidden_dim*2 , self.hidden_dim*2)
        self.fcIpd = nn.Linear(self.hidden_dim , self.hidden_dim)
        self.fcIpu = nn.Linear(self.hidden_dim , self.hidden_dim)
        self.fcIpdu = nn.Linear(self.hidden_dim*2 , self.hidden_dim*2)

        self.cemb=nn.Embedding(num_embeddings=self.cat_dim, embedding_dim=self.feature_dim, padding_idx=catPad)
        self.temb=nn.Embedding(num_embeddings=self.cat_dim+48, embedding_dim=self.feature_dim, padding_idx=catPad+48)
        self.uemb=nn.Embedding(num_embeddings=self.user_dim, embedding_dim=self.feature_dim)
        self.pemb=nn.Embedding(num_embeddings=self.poi_dim, embedding_dim=self.feature_dim, padding_idx=poiPad)
        self.softplus=nn.Softplus()
    
    def returnEmb(self, gtype, Num):
        if gtype=='cat':
            return self.cemb.weight[:Num]
        elif gtype=='time':
            return self.temb.weight[:Num]
        elif gtype=='poi':
            return self.pemb.weight[:Num]

    def BPRloss(self,emb,epos,eneg):
        pos=torch.bmm(emb.unsqueeze(dim=1), epos.unsqueeze(dim=2)).squeeze()
        neg=torch.bmm(emb.unsqueeze(dim=1), eneg.unsqueeze(dim=2)).squeeze()
        return self.softplus(neg-pos)
                
    def forward(self, u, cseq ,tseq, pseq, lastc):
        u=self.uemb(u)
    
        hc, (h_nc, c_nc) = self.lstmC(cseq)
        ht, (h_nt, c_nt) = self.lstmT(tseq) 

        #注意力機制
        ckey  =self.catkey(cseq)
        cquery=self.catquery(h_nc)
        tkey  =self.timekey(tseq)
        tquery=self.timequery(h_nt)
        cw=torch.matmul(ckey,torch.unsqueeze(cquery,-1))+torch.matmul(cseq,torch.unsqueeze(lastc,-1))
        tw=torch.matmul(tkey,torch.unsqueeze(tquery,-1))
        csoft=F.softmax(cw,dim=1)
        tsoft=F.softmax(tw,dim=1)
        crep=torch.sum(cseq*csoft,dim=1)
        trep=torch.sum(tseq*tsoft,dim=1)

        #類別預測輸出
        outputc=self.fcfuse(torch.cat([crep,trep],dim=1))
        outputc=F.dropout(outputc, self.dropout, training=self.training)
        outputc=self.fcfuse2(outputc)
        
        #解纏學習
        pc=self.fcPc(torch.mean(cseq,dim=1))
        pt=self.fcPt(torch.mean(tseq,dim=1))     
        pIc=self.fccrep(crep)   
        pIt=self.fctrep(trep)
        lcon=torch.sum(self.BPRloss(pIc,pc,pt)+self.BPRloss(pIt,pt,pc))
  
        #類別和時間表示融合
        crep1=self.fcIpc(crep)
        trep1=self.fcIpt(trep)
        ctrep=self.fcIpct(torch.cat([crep1,trep1],dim=1))
        ctrep=F.dropout(ctrep, self.dropout, training=self.training)

        #地理和使用者表示融合
        hp, (h_np, c_np) = self.lstmP(pseq)
        drep=self.fcIpd(h_np)
        urep=self.fcIpu(u)
        durep=self.fcIpdu(torch.cat([drep,urep],dim=1))
        durep=F.dropout(durep, self.dropout, training=self.training)

        #POI預測輸出
        Ip=self.fcIp1(torch.cat([ctrep,durep],dim=1))
        Ip=F.dropout(Ip, self.dropout, training=self.training)
        output=self.fcIp2(Ip)



        return outputc, output, lcon
