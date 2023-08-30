import torch.utils.data as data
from tqdm import tqdm, trange
from torch.nn.utils.rnn import pad_sequence
import pickle
import torch

#載入序列資料
#一筆序列對應欄位 : j['userid'],j['venueid'],j['categid'],j['latitute'],j['longitude'],j['hour'],j['hour_48'],j['week'],j['timestamp']
class myDataset(data.Dataset):
    def __init__(self, data_path, poiPad, catPad, timePad):
        xdata=[]
        cdata=[]
        tdata=[]
        self.udata=[]
        self.ydata=[]
        self.ydataC=[]
        self.lastc=[]
        f=open(data_path,'rb')
        data=pickle.load(f)
        print("data length:",len(data))
        
        for i in tqdm(data):
            self.ydata.append(i[-1][1])
            self.ydataC.append(i[-1][2])
            self.lastc.append(i[-2][2])
            self.udata.append(i[0][0])
            del i[-1]
            x=[]
            c=[]
            t=[]
            for j in range(len(i)):
                x.append(i[j][1])
                c.append(i[j][2])
                t.append(i[j][6])

            xdata.append(torch.tensor(x))
            cdata.append(torch.tensor(c))
            tdata.append(torch.tensor(t))


        self.xdata=pad_sequence(xdata, batch_first=True, padding_value=poiPad)
        self.cdata=pad_sequence(cdata, batch_first=True, padding_value=catPad)
        self.tdata=pad_sequence(tdata, batch_first=True, padding_value=timePad)

        
        self.n_samples =len(self.xdata)

    def __getitem__(self, index):
        return self.xdata[index], self.cdata[index], self.tdata[index], self.udata[index], self.ydata[index], self.ydataC[index], self.lastc[index]
    
    
    def __len__(self):
        return self.n_samples