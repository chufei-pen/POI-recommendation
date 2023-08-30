import numpy as np
import torch


#計算拉普拉斯矩陣
def calculate_laplacian_matrix(adj_mat):
    n_vertex = adj_mat.shape[0]

    # row sum
    #np.diag  將一維投射到二維的對角線  或是將二維的對角線保留到一維
    deg_mat_row = np.asmatrix(np.diag(np.sum(adj_mat, axis=1)))
    # column sum
    # deg_mat_col = np.asmatrix(np.diag(np.sum(adj_mat, axis=0)))
    deg_mat = deg_mat_row

    adj_mat = np.asmatrix(adj_mat)
    id_mat = np.asmatrix(np.identity(n_vertex))

    wid_deg_mat = deg_mat + id_mat
    wid_adj_mat = adj_mat + id_mat
    hat_rw_normd_lap_mat = np.matmul(np.linalg.matrix_power(wid_deg_mat, -1), wid_adj_mat)
    #np.linalg.matrix_power  陣列的值開次方
    
    return hat_rw_normd_lap_mat

#計算時間的拉普拉斯矩陣，額外計算相鄰時間的關係
def calculate_laplacian_matrix_time(adj_mat,catNum):
    n_vertex = adj_mat.shape[0]

    # row sum
    #np.diag  將一維投射到二維的對角線  或是將二維的對角線保留到一維
    deg_mat_row = np.asmatrix(np.diag(np.sum(adj_mat, axis=1)))
    # column sum
    # deg_mat_col = np.asmatrix(np.diag(np.sum(adj_mat, axis=0)))
    deg_mat = deg_mat_row

    adj_mat = np.asmatrix(adj_mat)
    id_mat = np.asmatrix(np.identity(n_vertex))

    id_mat_t = np.asmatrix(np.identity(48))
    id_mat_l = np.roll(id_mat_t,-1,1)
    id_mat_r = np.roll(id_mat_t,1,1)
    id_mat_l = np.pad(id_mat_l, ((0, catNum), (0, catNum)), mode='constant')
    id_mat_r = np.pad(id_mat_r, ((0, catNum), (0, catNum)), mode='constant')


    wid_deg_mat = deg_mat + id_mat*2
    wid_adj_mat = adj_mat + id_mat + id_mat_l*0.5 + id_mat_r*0.5
    hat_rw_normd_lap_mat = np.matmul(np.linalg.matrix_power(wid_deg_mat, -1), wid_adj_mat)
    #np.linalg.matrix_power  陣列的值開次方
    
    return hat_rw_normd_lap_mat


#將GCN輸出的每個嵌入帶入序列資料
def idx_to_emb(batch, all_poi_embeddings, pad_idx, device):
    all_batch=[]
    for seq in range(len(batch)):
        sequence=[]
        if pad_idx>1000:
            poi_embeddings=all_poi_embeddings
        else:
            poi_embeddings=all_poi_embeddings[seq]
        for idx in batch[seq]:
            if idx ==pad_idx:
                sequence.append(torch.zeros(64).to(device))
            else:
                poi_emb=poi_embeddings[idx]
                sequence.append(poi_emb)
        sequence=torch.stack(sequence)
        all_batch.append(sequence)
    all_batch=torch.stack(all_batch)
    return all_batch

#計算top K準確度
def topk(preds,targets,k):
    y_resize = targets.view(-1,1)
    _, pred = preds.topk(k, 1, True, True)
    correct = torch.eq(pred, y_resize).sum().float().item()
    return correct