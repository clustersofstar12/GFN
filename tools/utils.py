import torch
import torch.nn as nn
import torch.nn.functional as F

def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def knn(nsample, xyz, new_xyz):
    dist = square_distance(xyz, new_xyz)
    id = torch.topk(dist, k=nsample, dim=1, largest=False)[1]
    id = torch.transpose(id, 1, 2)  # [B, V, k]
    return id

class KNN_dist(nn.Module):
    def __init__(self, k):
        super(KNN_dist, self).__init__()
        '''self.R = nn.Sequential(
            nn.Linear(10, 10),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(10, 10),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(10, 1),
        )'''
        self.k=k
    def forward(self,F,vertices):
        id = knn(self.k, vertices, vertices)
        '''F = index_points(F, id)  # [B, V, K, C]
        v = index_points(vertices, id)
        v_0 = v[:, :, 0, :].unsqueeze(-2).repeat(1, 1, self.k, 1)
        v_F = torch.cat((v_0, v, v_0-v, torch.norm(v_0-v, dim=-1, p=2).unsqueeze(-1)), -1)
        v_F = self.R(v_F)  # [B, V, K, 1]
        F = torch.mul(v_F, F)
        F = torch.sum(F, -2)  # [B, V, C]
        v_F = v_F.squeeze(-1)
        v_F = v_F.index_select(0, torch.tensor([0]).cuda())
        v_F = v_F.squeeze(0)'''  # [V, K]
        id = id.index_select(0, torch.tensor([0]).cuda())
        id = id.squeeze(0)
        return id

'''class View_selector(nn.Module):
    def __init__(self, n_views, sampled_view):
        super(View_selector, self).__init__()
        self.n_views = n_views
        self.s_views = sampled_view
        self.cls = nn.Sequential(
            nn.Linear(512*self.s_views, 256*self.s_views),
            nn.LeakyReLU(0.2),
            nn.Linear(256*self.s_views, 40*self.s_views))
    def forward(self,F,vertices,k):
        id = farthest_point_sample(vertices,self.s_views)
        vertices1 = index_points(vertices,id)
        id_knn = knn(k,vertices,vertices1)
        F = index_points(F,id_knn)
        vertices = index_points(vertices,id_knn)
        F1 = F.transpose(1,2).reshape(F.shape[0],k,self.s_views*F.shape[-1])
        F_score = self.cls(F1).reshape(F.shape[0],k,self.s_views,40).transpose(1,2)
        F1_ = Functional.softmax(F_score,-3)
        F1_ = torch.max(F1_,-1)[0]
        F1_id = torch.argmax(F1_,-1)
        F1_id = Functional.one_hot(F1_id,4).float()
        F1_id_v = F1_id.unsqueeze(-1).repeat(1,1,1,3)
        F1_id_F = F1_id.unsqueeze(-1).repeat(1, 1, 1, 512)
        F_new = torch.mul(F1_id_F,F).sum(-2)
        vertices_new = torch.mul(F1_id_v,vertices).sum(-2)
        return F_new,F_score,vertices_new'''

class LocalGCN(nn.Module):
    def __init__(self, k, n_views):
        super(LocalGCN, self).__init__()
        '''self.conv = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )'''
        self.k = k
        self.n_views = n_views
        self.KNN = KNN_dist(k=self.k)
    def forward(self, F, V):
        id = self.KNN(F, V)
        '''F = F.view(-1, 512)
        F = self.conv(F)
        F = F.view(-1, self.n_views, 512)'''
        return id

class GCN(nn.Module):
    def __init__(self, bias, n_view):
        super(GCN, self).__init__()

        self.w = nn.Parameter(torch.empty(size=(512, 512)))
        nn.init.xavier_uniform_(self.w.data, gain=1.414)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(512).cuda())
        else:
            self.bias = None

        self.fusion = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),)
        self.n_views = n_view


    def forward(self, f, adj):
        f = torch.matmul(adj, f)  # [B, k, C]
        f = torch.matmul(f, self.w)  # [B, k, C]
        if self.bias is not None:
            f = f + self.bias
        #print(f.size())
        f = f.view(-1, 512)
        f = self.fusion(f)  # [B, k, C]
        f = f.view(-1, self.n_views, 512)
        return f


class NonLocalMP(nn.Module):
    def __init__(self, n_view):
        super(NonLocalMP, self).__init__()
        self.n_view = n_view
        self.Relation = nn.Sequential(
            nn.Linear(2 * 512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.Fusion = nn.Sequential(
            nn.Linear(2 * 512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, F):
        F_i = torch.unsqueeze(F, 2)
        F_j = torch.unsqueeze(F, 1)
        F_i = F_i.repeat(1, 1, self.n_view, 1)
        F_j = F_j.repeat(1, self.n_view, 1, 1)
        M = torch.cat((F_i, F_j), 3)
        M = self.Relation(M)
        M = torch.sum(M,-2)
        F = torch.cat((F, M), 2)
        F = F.view(-1, 512 * 2)
        F = self.Fusion(F)
        F = F.view(-1, self.n_view, 512)
        return F

'''class AssignMatrix(nn.Module):
    def __init__(self, kc, n_view):
        super(AssignMatrix, self).__init__()
        self.Slearn = nn.Sequential(
            nn.Linear(1024, 512),
            #nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            #nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, kc),
            nn.BatchNorm1d(kc),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.kc = kc
        self.n_view = n_view

    def forward(self, f, adj):
        #print(f.size())
        #print(adj.size())
        f1 = torch.matmul(adj, f)  # [B, V, C]
        f2 = torch.cat((f, f1), 2)  # [B, V, 2C]
        f2 = f2.view(-1, 2*512)
        #f = f.view(-1, 512)
        s = self.Slearn(f2)
        s = F.softmax(s, dim=-1)
        s = s.view(-1, self.n_view, self.kc)  # [B, V, K]
        return s

class AssignMat(nn.Module):
    def __init__(self, kc, n_view):
        super(AssignMat, self).__init__()
        self.Slearn = nn.Sequential(
            nn.Linear(1024, 512),
            #nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Linear(512, 512),
            #nn.BatchNorm1d(512),
            #nn.LeakyReLU(0.2, inplace=True),
            #nn.Linear(512, 512),
            #nn.BatchNorm1d(512),
            #nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            #nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, kc),
            nn.BatchNorm1d(kc),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.kc = kc
        self.n_view = n_view

    def forward(self, f, adj):
        #print(f.size())
        #print(adj.size())
        f1 = torch.matmul(adj, f)  # [B, V, C]
        f2 = torch.cat((f, f1), 2)  # [B, V, 2C]
        f2 = f2.view(-1, 2*512)
        #f = f.view(-1, 512)
        s = self.Slearn(f2)
        s = F.softmax(s, dim=-1)
        s = s.view(-1, self.n_view, self.kc)  # [B, V, K]
        return s'''

'''class Assign(nn.Module):
    def __init__(self, kc, n_view):
        super(Assign, self).__init__()
        self.w = nn.Parameter(torch.empty(size=(512, kc)))
        nn.init.xavier_uniform_(self.w.data, gain=1.414)
        self.bias = nn.Parameter(torch.FloatTensor(kc).cuda())
        self.fu = nn.Sequential(
            nn.BatchNorm1d(kc),
            nn.LeakyReLU(0.2),)
        self.kc = kc
        self.n_view = n_view

    def forward(self, f, adj):
        f1 = torch.matmul(adj, f)
        s = torch.matmul(f1, self.w)  # [B, k, k1]
        s = s + self.bias
        s = s.view(-1, self.kc)
        s = self.fu(s)
        s = F.softmax(s, dim=-1)
        s = s.view(-1, self.n_view, self.kc)
        return s'''


class AttentionLayer(nn.Module):
    def __init__(self):
        super(AttentionLayer, self).__init__()

        self.dropout = 0.2
        #self.in_features = in_features
        #self.out_features = out_features

        self.w = nn.Parameter(torch.empty(size=(512, 512)))
        nn.init.xavier_uniform_(self.w.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * 512, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.out_features = 512

    def forward(self, x):
        B, N, C = x.size()

        h = torch.matmul(x, self.w)
        h = torch.cat([h.repeat(1, 1, N).view(B, N * N, C), h.repeat(1, N, 1)], dim=2).view(B, N, N,
                2 * self.out_features)  # [B,N,N,2C]
        attention = self.leakyrelu(torch.matmul(h, self.a).squeeze(3))  # [B,N,N]
        attention = F.softmax(attention, dim=2)  # [B,N,N]
        print('attention', attention)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, x)  # [B,N,C]
        #out = F.elu(h_prime + self.beta * h)

        return h_prime

class AttentionMP(nn.Module):
    def __init__(self, dropout):
        super(AttentionMP, self).__init__()
        nheads = 10
        self.dropout = dropout
        self.attentions = [AttentionLayer() for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        #self.out_att = AttentionLayer()

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        #x = torch.cat([att(x) for att in self.attentions], dim=1)
        x = self.attention_0(x) + self.attention_1(x) + self.attention_2(x) + self.attention_3(x) + self.attention_4(x)
        x = x + self.attention_5(x) + self.attention_6(x) + self.attention_7(x) + self.attention_8(x) + self.attention_9(x)
        x = torch.div(x, 10.0)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(x)
        x = F.log_softmax(x, dim=1)
        return x

